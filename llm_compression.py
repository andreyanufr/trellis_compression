import torch
from transformers import AutoModelForCausalLM
from trellis import trellis_encode, trellis_decode
from trellis import trellis_encode_4_bit, trellis_decode_4_bit
from trellis import trellis_encode_bit_shift, trellis_decode_bit_shift



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
    assert num_bits in (2, 4, 8)
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

def encode_decode_trellis(data, scramble_bits=True, with_group=False, num_bits=8):
    _, compressed_data, scale, zero_point = encode_decode_asym(data, num_bits=num_bits, return_encoded_data=True)

    if with_group:
        group_size = 32
        n_chunks = compressed_data.size(0) // group_size

        decoded_data = []

        for i in range(n_chunks):
            chunk = compressed_data[i*group_size:(i+1)*group_size]
            if num_bits == 4:
                encoded_chunk = trellis_encode_4_bit(chunk, scramble_bits=scramble_bits)
                decoded_chunk = trellis_decode_4_bit(encoded_chunk, N=chunk.size(0), descramble_bits=scramble_bits)
            else:
                encoded_chunk = trellis_encode(chunk, scramble_bits=scramble_bits)
                decoded_chunk = trellis_decode(encoded_chunk, N=chunk.size(0), descramble_bits=scramble_bits)
            decoded_data.append(decoded_chunk)
        decoded_data = torch.cat(decoded_data, dim=0)
    else:
        if num_bits == 4:
            encoded_data = trellis_encode_4_bit(compressed_data, scramble_bits=scramble_bits)
            decoded_data = trellis_decode_4_bit(encoded_data, N=compressed_data.size(0), descramble_bits=scramble_bits)
        else:
            encoded_data = trellis_encode(compressed_data, scramble_bits=scramble_bits)
            decoded_data = trellis_decode(encoded_data, N=compressed_data.size(0), descramble_bits=scramble_bits)

    #print(f"Asymmetric trellis {8}-bit: {decoded_data}")

    decompressed_data = asymmetric_dequantize(decoded_data, scale, zero_point)
    return decompressed_data


def encode_decode_trellis_8_to_2(data, scramble_bits=True, device=None):
    _, compressed_data, scale, zero_point = encode_decode_asym(data, num_bits=8, return_encoded_data=True)

    if len(compressed_data.shape) == 2:

        n_chunks = compressed_data.size(0)
        decoded_data = []

        if device is not None:
            compressed_data = compressed_data.to(device)
        for i in range(n_chunks):
            chunk = compressed_data[i, :]
            encoded_chunk = trellis_encode_bit_shift(chunk, scramble_bits=scramble_bits, k=2)
            decoded_chunk = trellis_decode_bit_shift(encoded_chunk, N=chunk.size(0), descramble_bits=scramble_bits, k=2)
            decoded_data.append(decoded_chunk)
        decoded_data = torch.cat(decoded_data, dim=0).to(compressed_data.device)
    else:
        encoded_data = trellis_encode_bit_shift(compressed_data, scramble_bits=scramble_bits, k=2)
        decoded_data = trellis_decode_bit_shift(encoded_data, N=compressed_data.size(0), descramble_bits=scramble_bits, k=2)


    #print(f"Asymmetric trellis {8}-bit: {decoded_data}")

    decompressed_data = asymmetric_dequantize(decoded_data, scale, zero_point)
    return decompressed_data



model_id = "meta-llama/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cpu')


#weight = model.model.layers[0].self_attn.q_proj.weight.data.clone()
if __name__ == "__main__":
    for n_layer in range(10):
        weight = model.model.layers[n_layer].mlp.down_proj.weight.data.clone()
        print(f"Layer {n_layer} down_proj weight compression results:")
        for i in range(3):
            row = weight[10 * i].clone()[:128]

            if n_layer == 0 and i == 0:
                row = torch.tensor([0.0378418 ,  0.08740234, -0.02368164, -0.10058594,  0.00723267,
                                    -0.10302734, -0.01556396,  0.03039551, -0.0027771 ,  0.01416016,
                                        0.02893066, -0.01483154, -0.55078125, -0.0007515 , -0.02954102,
                                        0.04394531,  0.01031494,  0.012146  , -0.02282715,  0.03564453,
                                        0.05664062,  0.03686523,  0.03930664, -0.02905273,  0.04272461,
                                        0.05444336,  0.06103516, -0.03662109,  0.02807617,  0.01757812,
                                    -0.04052734, -0.01647949], dtype=torch.float32)


            row_trellis_8_to_2 = encode_decode_trellis_8_to_2(row, scramble_bits=False)
            row_trellis_8_to_2_scrambled = encode_decode_trellis_8_to_2(row, scramble_bits=True)

            row_2bit = encode_decode_asym(row, num_bits=2)
            row_4bit = encode_decode_asym(row, num_bits=4)
            row_8bit = encode_decode_asym(row, num_bits=8)

            row_trellis = encode_decode_trellis(row, with_group=True, scramble_bits=False)
            row_trellis_scrambled = encode_decode_trellis(row, scramble_bits=True)


            row_trellis_4_bit = encode_decode_trellis(row, scramble_bits=False, num_bits=4)
            row_trellis_scrambled_4_bit = encode_decode_trellis(row, scramble_bits=True, num_bits=4)


            res = {
                "8bit             ": row_8bit,
                "4bit             ": row_4bit,
                "trellis_8_to_4   ": row_trellis,
                "trellis_sc_8_to_4": row_trellis_scrambled,
                "2bit             ": row_2bit,
                "trellis_4_to_2   ": row_trellis_4_bit,
                "trellis_sc_4_to_2": row_trellis_scrambled_4_bit,
                "trellis_8_to_2   ": row_trellis_8_to_2,
                "trellis_sc_8_to_2": row_trellis_8_to_2_scrambled
            }

            print("_" * 80)
            for key, value in res.items():
                dispersion = torch.std(row.float() - value.float()).item()
                print(f"\t{key}: Dispersion = {dispersion}")
            for key, value in res.items():
                relative_error = torch.norm(row.float() - value.float()) / torch.norm(row.float())
                print(f"\t{key}: Relative Error = {relative_error:.6f}")




