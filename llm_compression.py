import torch
from transformers import AutoModelForCausalLM
from trellis import trellis_encode, trellis_decode
from trellis import trellis_encode_4_bit, trellis_decode_4_bit



def asymmetric_quantize(x: torch.Tensor, num_bits: int):
    """
    Asymmetric per-tensor quantization.

    Args:
        x (torch.Tensor): float tensor
        num_bits (int): 8 or 4

    Returns:
        q (torch.Tensor): quantized int tensor
        scale (float): quantization scale
        zero_point (int): zero point
    """
    assert num_bits in (4, 8)
    assert x.is_floating_point()

    qmin = 0
    qmax = (1 << num_bits) - 1

    x_min = x.min()
    x_max = x.max()

    # Avoid degenerate case
    if x_min == x_max:
        scale = 1.0
        zero_point = 0
        q = torch.zeros_like(x, dtype=torch.uint8)
        return q, scale, zero_point

    scale = (x_max - x_min) / float(qmax - qmin)

    # zero_point computed from min value
    zero_point = qmin - torch.round(x_min / scale)
    zero_point = int(torch.clamp(zero_point, qmin, qmax).item())

    # Quantize
    q = torch.round(x / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)

    # Choose smallest integer dtype that fits
    if num_bits == 8:
        q = q.to(torch.uint8)
    else:
        q = q.to(torch.int8)  # values still in [0, 15]

    return q, float(scale), zero_point


def asymmetric_dequantize(q: torch.Tensor, scale: float, zero_point: int):
    return scale * (q.float() - zero_point)


def encode_decode_asym(data, num_bits, return_encoded_data=False):
    compressed_data, scale, zero_point = asymmetric_quantize(data, num_bits=num_bits)
    decompressed_data = asymmetric_dequantize(compressed_data, scale, zero_point)

    if return_encoded_data:
        return decompressed_data, compressed_data, scale, zero_point
    #print(f"Asymmetric {num_bits}-bit: {compressed_data}")
    return decompressed_data

def encode_decode_trellis(data, scramble_bits=True, with_group=False):
    _, compressed_data, scale, zero_point = encode_decode_asym(data, num_bits=8, return_encoded_data=True)
    
    if with_group:
        group_size = 32
        n_chunks = compressed_data.size(0) // group_size
        
        decoded_data = []
        
        for i in range(n_chunks):
            chunk = compressed_data[i*group_size:(i+1)*group_size]
            encoded_chunk = trellis_encode(chunk, scramble_bits=scramble_bits)
            decoded_chunk = trellis_decode(encoded_chunk, N=chunk.size(0), descramble_bits=scramble_bits)
            decoded_data.append(decoded_chunk)
        decoded_data = torch.cat(decoded_data, dim=0)
    else:        
        encoded_data = trellis_encode(compressed_data, scramble_bits=scramble_bits)
        decoded_data = trellis_decode(encoded_data, N=compressed_data.size(0), descramble_bits=scramble_bits)
    
    #print(f"Asymmetric trellis {8}-bit: {decoded_data}")

    decompressed_data = asymmetric_dequantize(decoded_data, scale, zero_point)
    return decompressed_data



model_id = "meta-llama/Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cpu')


#weight = model.model.layers[0].self_attn.q_proj.weight.data.clone()

for n_layer in range(32):
    weight = model.model.layers[n_layer].mlp.down_proj.weight.data.clone()
    print(f"Layer {n_layer} down_proj weight compression results:")
    for i in range(3):
        row = weight[10 * i].clone()


        row_4bit = encode_decode_asym(row, num_bits=4)
        row_8bit = encode_decode_asym(row, num_bits=8)

        row_trellis = encode_decode_trellis(row, scramble_bits=False)
        row_trellis_scrambled = encode_decode_trellis(row, scramble_bits=True)

        res = {
            "8bit             ": row_8bit,
            "4bit             ": row_4bit,
            "trellis          ": row_trellis,
            "trellis_scrambled": row_trellis_scrambled
        }

        print("_" * 80)
        for key, value in res.items():
            dispersion = torch.std(row.float() - value.float()).item()
            print(f"\t{key}: Dispersion = {dispersion}")




