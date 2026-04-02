import torch

def unpack_with_overlap(packed_tensor: torch.Tensor, overlap: int, num_elements: int) -> torch.Tensor:
    """
    Unpacks a uint8 tensor by reconstructing overlapping bits.
    """
    if not 1 <= overlap <= 7:
        raise ValueError("Overlap must be between 1 and 7")
    
    output = torch.zeros(num_elements, dtype=torch.uint8)
    stride = 8 - overlap
    
    # We treat the packed tensor as a long bit stream
    # To get the i-th element, we need to find where its 8 bits start
    # Element 0 starts at bit 0
    # Element 1 starts at bit (1 * stride)
    # Element 2 starts at bit (2 * stride)
    
    for i in range(num_elements):
        start_bit = i * stride
        byte_idx = start_bit // 8
        bit_offset = start_bit % 8
        
        # Pull the first part from the current byte
        # We need to shift left to clear high bits, then shift right to align
        current_byte = packed_tensor[byte_idx].item()
        
        # Mask to get rid of bits before our start_bit in the current byte
        # and shift them to be the high bits of our new byte
        bits_available_in_first_byte = 8 - bit_offset
        
        if bits_available_in_first_byte >= 8:
            # The byte aligns perfectly (only happens at index 0 or if stride is 8)
            res = current_byte
        else:
            # Take the remaining bits from the current byte
            mask = (1 << bits_available_in_first_byte) - 1
            res = (current_byte & mask) << (8 - bits_available_in_first_byte)
            
            # If we haven't filled all 8 bits, grab the rest from the next byte
            bits_needed = 8 - bits_available_in_first_byte
            if bits_needed > 0 and (byte_idx + 1) < len(packed_tensor):
                next_byte = packed_tensor[byte_idx + 1].item()
                # Take the top 'bits_needed' from the next byte
                res |= (next_byte >> (8 - bits_needed))
        
        output[i] = res
        
    return output



def pack_with_overlap(input_tensor: torch.Tensor, overlap: int) -> torch.Tensor:
    """
    Packs a uint8 tensor by removing redundant overlapping bits between 
    consecutive elements.
    """
    if not 1 <= overlap <= 7:
        raise ValueError("Overlap must be between 1 and 7")

    # Number of new unique bits each subsequent byte provides
    stride = 8 - overlap
    
    # Calculate total bits: first byte (8) + remaining bytes * stride
    total_bits = 8 + (len(input_tensor) - 1) * stride
    # Calculate required output length (round up to nearest byte)
    output_len = (total_bits + 7) // 8
    
    output = torch.zeros(output_len, dtype=torch.uint8)
    
    current_bit_offset = 0
    
    for i, value in enumerate(input_tensor):
        val = value.item()
        
        if i == 0:
            # First byte is always fully copied
            output[0] = val
            current_bit_offset = 8
        else:
            # Mask out the overlapping high bits
            # Example: if overlap is 4, we want to keep only the lower 4 bits
            # Mask: (1 << 4) - 1 = 0b00001111
            mask = (1 << stride) - 1
            unique_bits = val & mask
            
            # Place the unique bits into the output buffer
            # They might span across two different bytes in the output
            target_byte_idx = current_bit_offset // 8
            bit_in_byte_offset = current_bit_offset % 8
            
            # Shift unique bits to the left to align with the current offset
            # and OR them into the output
            if bit_in_byte_offset + stride <= 8:
                # Fits within the current byte
                shift = 8 - bit_in_byte_offset - stride
                output[target_byte_idx] |= (unique_bits << shift)
            else:
                # Spans across two bytes
                high_part_len = 8 - bit_in_byte_offset
                low_part_len = stride - high_part_len
                
                output[target_byte_idx] |= (unique_bits >> low_part_len)
                output[target_byte_idx + 1] |= ((unique_bits & ((1 << low_part_len) - 1)) << (8 - low_part_len))
            
            current_bit_offset += stride

    return output


if __name__ == "__main__":
    # --- Example Usage ---
    # 0b01011010 (90), 0b10101000 (168), 0b10001111 (143)
    # Overlap 4: 
    # 1. 0101 [1010]
    # 2. [1010] 1000 -> take '1000'
    # 3. [1000] 1111 -> take '1111'
    # Result: 01011010 11110000 (if padded) -> [90, 240] in our example [0b01011010, 0b10001111]
    input_data = torch.tensor([90, 168, 143], dtype=torch.uint8)
    packed = pack_with_overlap(input_data, 4)
    print("Original Tensor:", input_data)
    print(f"Packed Tensor: {packed}")
    print(f"Binary: {[bin(b.item()) for b in packed]}")
    unpacked = unpack_with_overlap(packed, 4, len(input_data))
    print(f"Unpacked Tensor: {unpacked}")
    
    
    input_data = torch.tensor([0b10101111, 0b10111100, 0b11110010, 0b11001011, 0b00101101, 0b10110100], dtype=torch.uint8)
    packed = pack_with_overlap(input_data, 6)
    print("Original Tensor:", input_data)
    print(f"Packed Tensor: {packed}")
    print(f"Binary: {[bin(b.item()) for b in packed]}")
    unpacked = unpack_with_overlap(packed, 6, len(input_data))
    print(f"Unpacked Tensor: {unpacked}")
