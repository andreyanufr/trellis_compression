import torch
import torch.nn as nn
import torch.nn.functional as F



def viterbi_gpt(states, init, trans, emit, obs):
    """
    Viterbi algorithm.

    Args:
        states (int): number of hidden states S
        init (Tensor): shape (S,), initial state probabilities
        trans (Tensor): shape (S, S), transition probabilities
        emit (Tensor): shape (S, O), emission probabilities
        obs (Tensor or list): length T, observation indices

    Returns:
        path (Tensor): shape (T,), most likely state sequence
    """
    T = len(obs)

    # prob[t, s] = max probability of being in state s at time t
    prob = torch.zeros(T, states, dtype=init.dtype)

    # prev[t, s] = previous state leading to state s at time t
    prev = torch.zeros(T, states, dtype=torch.long)

    # Initialization (t = 0)
    for s in range(states):
        prob[0, s] = init[s] * emit[s, 0]

    # Dynamic programming
    for t in range(1, T):
        for s in range(states):
            best_prob = 0.0
            best_state = 0
            for r in range(states):
                new_prob = prob[t - 1, r] * trans[r, s] * emit[s, t]
                if new_prob > best_prob:
                    best_prob = new_prob
                    best_state = r
            prob[t, s] = best_prob
            prev[t, s] = best_state

    # Backtracking
    path = torch.zeros(T, dtype=torch.long)

    # Last state
    path[T - 1] = torch.argmax(prob[T - 1])

    # Follow backpointers
    for t in range(T - 2, -1, -1):
        path[t] = prev[t + 1, path[t + 1]]

    return path


# from https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm
def viterbi(y, A, B, Pi=None):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
    B : array (K, T)
        Emission matrix. See HiddenMarkovModel.emission for details.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else torch.full((K,), 1 / K)
    T = len(y)
    T1 = torch.empty((K, T), dtype=torch.float64)
    T2 = torch.empty((K, T), dtype=torch.uint8)

    # Initilaize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, y[0]]
    T2[:, 0] = 0

    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = torch.max(T1[:, i - 1] * A.T * B[torch.newaxis, :, y[i]].T, 1)
        T2[:, i] = torch.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = torch.empty(T, dtype=torch.uint8)
    x[-1] = torch.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2


def viterbi_path(prior, transmat, obslik, scaled=True, ret_loglik=False):
    '''Finds the most-probable (Viterbi) path through the HMM state trellis
    Notation:
        Z[t] := Observation at time t
        Q[t] := Hidden state at time t
    Inputs:
        prior: np.array(num_hid)
            prior[i] := Pr(Q[0] == i)
        transmat: np.ndarray((num_hid,num_hid))
            transmat[i,j] := Pr(Q[t+1] == j | Q[t] == i)
        obslik: np.ndarray((num_hid,num_obs))
            obslik[i,t] := Pr(Z[t] | Q[t] == i)
        scaled: bool
            whether or not to normalize the probability trellis along the way
            doing so prevents underflow by repeated multiplications of probabilities
        ret_loglik: bool
            whether or not to return the log-likelihood of the best path
    Outputs:
        path: np.array(num_obs)
            path[t] := Q[t]
    '''
    num_hid = obslik.shape[0] # number of hidden states
    num_obs = obslik.shape[1] # number of observations (not observation *states*)

    # trellis_prob[i,t] := Pr((best sequence of length t-1 goes to state i), Z[1:(t+1)])
    trellis_prob = torch.zeros((num_hid,num_obs))
    # trellis_state[i,t] := best predecessor state given that we ended up in state i at t
    trellis_state = torch.zeros((num_hid,num_obs), dtype=torch.int64) # int because its elements will be used as indicies
    path = torch.zeros(num_obs, dtype=torch.int64) # int because its elements will be used as indicies

    trellis_prob[:,0] = prior * obslik[:,0] # element-wise mult
    if scaled:
        scale = torch.ones(num_obs) # only instantiated if necessary to save memory
        scale[0] = 1.0 / torch.sum(trellis_prob[:,0])
        trellis_prob[:,0] *= scale[0]

    trellis_state[:,0] = 0 # arbitrary value since t == 0 has no predecessor
    for t in range(1, num_obs):
        for j in range(num_hid):
            trans_probs = trellis_prob[:,t-1] * transmat[:,j] # element-wise mult
            trellis_state[j,t] = trans_probs.argmax()
            trellis_prob[j,t] = trans_probs[trellis_state[j,t]] # max of trans_probs
            trellis_prob[j,t] *= obslik[j,t]
        if scaled:
            scale[t] = 1.0 / torch.sum(trellis_prob[:,t])
            trellis_prob[:,t] *= scale[t]

        # if torch.sum(trellis_prob[:, t]) < 0.00001:
        #     # re-scale to avoid numerical issues
        #     trellis_prob[:, t] = trellis_prob[:, t] + 0.0001
        #     if scaled:
        #         scale[t] = 1.0 / torch.sum(trellis_prob[:, t])
        #         trellis_prob[:, t] *= scale[t]


    path[-1] = trellis_prob[:,-1].argmax()
    for t in range(num_obs-2, -1, -1):
        path[t] = trellis_state[(path[t+1]), t+1]

    if not ret_loglik:
        return path
    else:
        if scaled:
            loglik = -torch.sum(torch.log(scale))
        else:
            p = trellis_prob[path[-1],-1]
            loglik = torch.log(p)
        return path, loglik


def trellis_encode(input_bits, scramble_bits=False):
    """
    Encodes input sequence of 8bit numbers using viterbi trellis encoding.
    One 8bit number have output edge to other if last 4 bits of current number
    are same as first 4 bits of next number. The output is a sequence of 8bit numbers this two times less elements.


    Args:
        input_bits (torch.Tensor): A tensor of shape (N) where N is the number of 8bit numbers.
        scramble_bits (bool): If True, scramble the input bits before encoding. Default is False.
    Returns:
        torch.Tensor: A tensor of shape (N / 2 + 1).
    """
    N = input_bits.size(0)
    N_OUT = N // 2 + 1

    # if scramble_bits:
    #     input_bits = ((input_bits >> 3) | (input_bits << 5))


    assert input_bits.dtype in [torch.uint8, torch.int8], "Input bits must be of type 8 bit type"
    if input_bits.dtype == torch.int8:
        input_bits = input_bits.to(torch.uint8)


    transitions = torch.zeros((256, 256), dtype=torch.float32)

    for i in range(256):
        for j in range(256):
            if (i & 0x0F) == (j >> 4):
                transitions[i, j] = 1.0 / 16.0

    states = torch.arange(256, dtype=torch.uint8)
    if scramble_bits:
        states = ((states >> 3) | (states << 5)) & 0xFF

        for i in range(256):
            if not i in states:
                raise ValueError("Scrambling resulted in non-unique states")

    states = states.unsqueeze(1).float()

    distance = (states - input_bits.unsqueeze(0).float())**2
    distance = torch.exp(-distance / (2 * (16.0 ** 2)))
    distance = distance / distance.sum(dim=0, keepdim=True)


    path = viterbi_path(
        prior=distance[:, 0],
        transmat=transitions,
        obslik=distance,
        scaled=True,
        ret_loglik=False
    )

    encoded_bits = torch.zeros((N_OUT,), dtype=torch.uint8)
    encoded_bits[0] = path[0]
    for i in range(N_OUT):
        if 2 * i < N:
            encoded_bits[i] = path[2 * i]

    if N % 2 == 0:
        encoded_bits[-1] = path[-1] << 4


    return encoded_bits


def trellis_decode(encoded_bits, N, descramble_bits=False):
    """
    Decodes input sequence of 8bit numbers using viterbi trellis decoding.
    One 8bit number have output edge to other if last 4 bits of current number
    are same as first 4 bits of next number. The input is a sequence of 8bit numbers
    this two times less elements.


    Args:
        encoded_bits (torch.Tensor): A tensor of shape (N // 2 + 1) where N is the number of 8bit numbers.
        N (int): The number of output 8bit numbers after decoding.
        descramble_bits (bool): If True, descramble the output bits after decoding. Default is False.
    Returns:
        torch.Tensor: A tensor of shape (N).
    """
    res = torch.zeros(N, dtype=torch.uint8)
    for i in range(encoded_bits.size(0)):
        if i * 2 >= N:
            break
        res[i * 2] = encoded_bits[i]
        if i > 0:
            res[i * 2 - 1] = ((encoded_bits[i - 1] & 0x0F) << 4) | (encoded_bits[i] >> 4)

    if N % 2 == 0:
        res[-1] = (encoded_bits[-1] >> 4) | ((encoded_bits[-2] & 0x0F) << 4)


    if descramble_bits:
        res = ((res >> 3) | (res << 5)) & 0xFF

    return res



def trellis_encode_4_bit(input_bits, scramble_bits=False):
    """
    Encodes input sequence of 4bit numbers using viterbi trellis encoding.
    One 4bit number have output edge to other if last 2 bits of current number
    are same as first 2 bits of next number. The output is a sequence of 8bit numbers this two times less elements.


    Args:
        input_bits (torch.Tensor): A tensor of shape (N) where N is the number of 4bit numbers.
        scramble_bits (bool): If True, scramble the input bits before encoding. Default is False.
    Returns:
        torch.Tensor: A tensor of shape (N / 2 + 1).
    """
    N = input_bits.size(0)
    N_OUT = N // 2 + 1

    # if scramble_bits:
    #     input_bits = ((input_bits >> 3) | (input_bits << 5))


    assert input_bits.dtype in [torch.uint8, torch.int8], "Input bits must be of type 8 bit type"
    if input_bits.dtype == torch.int8:
        input_bits = input_bits.to(torch.uint8)


    transitions = torch.zeros((16, 16), dtype=torch.float32)

    for i in range(16):
        for j in range(16):
            if (i & 0x03) == (j >> 2):
                transitions[i, j] = 1.0 / 4.0

    states = torch.arange(16, dtype=torch.uint8)
    if scramble_bits:
        states = ((states >> 1) | (states << 3)) & 0b00001111

        for i in range(16):
            if not i in states:
                raise ValueError("Scrambling resulted in non-unique states")

    states = states.unsqueeze(1).float()

    distance = (states - input_bits.unsqueeze(0).float())**2
    distance = torch.exp(-distance / (2 * (4.0 ** 2)))

    distance = distance**2 # sharpening the distribution to make it more deterministic, since we have less states and more collisions

    distance = distance / distance.sum(dim=0, keepdim=True)


    path = viterbi_path(
        prior=distance[:, 0],
        transmat=transitions,
        obslik=distance,
        scaled=True,
        ret_loglik=False
    )

    # path = viterbi_gpt(
    #     states=256,
    #     init=distance[:, 0],
    #     trans=transitions,
    #     emit=distance,
    #     obs=input_bits
    # )

    encoded_bits = torch.zeros((N_OUT,), dtype=torch.uint8)
    encoded_bits[0] = path[0]
    for i in range(N_OUT):
        if 2 * i < N:
            encoded_bits[i] = path[2 * i]

    if N % 2 == 0:
        encoded_bits[-1] = path[-1]#input_bits[-1]#path[-1] << 2


    return encoded_bits


def trellis_decode_4_bit(encoded_bits, N, descramble_bits=False):
    """
    Decodes input sequence of 8bit numbers using viterbi trellis decoding.
    One 8bit number have output edge to other if last 4 bits of current number
    are same as first 4 bits of next number. The input is a sequence of 8bit numbers
    this two times less elements.


    Args:
        encoded_bits (torch.Tensor): A tensor of shape (N // 2 + 1) where N is the number of 8bit numbers.
        N (int): The number of output 8bit numbers after decoding.
        descramble_bits (bool): If True, descramble the output bits after decoding. Default is False.
    Returns:
        torch.Tensor: A tensor of shape (N).
    """
    res = torch.zeros(N, dtype=torch.uint8)
    for i in range(encoded_bits.size(0)):
        if i * 2 >= N:
            break
        res[i * 2] = encoded_bits[i]
        if i > 0:
            res[i * 2 - 1] = ((encoded_bits[i - 1] & 0x03) << 2) | (encoded_bits[i] >> 2)

    if N % 2 == 0:
        res[-1] = encoded_bits[-1]  #((encoded_bits[-1] >> 2) | ((encoded_bits[-2] & 0x03) << 2))


    if descramble_bits:
        res = ((res >> 1) | (res << 3)) & 0b00001111

    return res


def pack(input_data, shift):
    """
    Pack uint8 tensor by combining bits from consecutive elements.

    Args:
        input_data (torch.Tensor): Input tensor of type torch.uint8
        shift (int): Number of bits to share between consecutive elements (1-7)

    Returns:
        torch.Tensor: Packed tensor (flattened, smaller than input)

    Logic:
        Elements at indices i and i+1 share 'shift' bits.
        The lower 'shift' bits of element i equal the upper 'shift' bits of element i+1.
        We store: all 8 bits of element 0, then only the lower (8-shift) bits of each
        subsequent element, since their upper shift bits are redundant.
    """
    if shift <= 0 or shift >= 8:
        raise ValueError("shift must be between 1 and 7")
    assert input_data.dtype == torch.uint8, "Input data must be of type torch.uint8"

    # Flatten input if not already flat
    input_flat = input_data.flatten()
    n = input_flat.size(0)

    if n == 0:
        return torch.tensor([], dtype=torch.uint8)

    if n == 1:
        return input_flat.clone()

    # Calculate total bits needed: 8 + (n-1)*(8-shift)
    total_bits = shift + (n-1)*(8-shift)
    num_bytes = (total_bits + 7) // 8  # Round up

    # Create a bit buffer
    packed = torch.zeros(num_bytes, dtype=torch.uint8)

    if 8 - shift == 2:
        for i in range(num_bytes):
            if i * 4  < n:
                packed[i] = input_flat[i * 4]
        if n > 0 and (n - 1) % 4 != 0:
            n_bits  = ((n - 1) % 4) * 2
            packed[-1] = input_flat[-1] << (8 - n_bits)

        return packed

    bit_pos = 0  # Current bit position in the output

    # Store all 8 bits of first element
    packed[0] = input_flat[0]

    # Store lower (8-shift) bits of remaining elements
    for i in range(1, n):
        elem = input_flat[i].item()
        for bit_idx in range(8 - shift):  # Only lower (8-shift) bits
            byte_idx = bit_pos // 8
            bit_in_byte = bit_pos % 8
            bit_value = (elem >> bit_idx) & 1
            packed[byte_idx] = packed[byte_idx] | (bit_value << bit_in_byte)
            bit_pos += 1

    return packed


def unpack(packed_data, shift, original_size):
    """
    Unpack a tensor that was packed using the pack function.

    Args:
        packed_data (torch.Tensor): Packed tensor of type torch.uint8
        shift (int): Number of bits that were shared (same as used in pack)
        original_size (int): Original number of elements before packing

    Returns:
        torch.Tensor: Unpacked tensor of shape (original_size,) with dtype torch.uint8
    """
    if shift <= 0 or shift >= 8:
        raise ValueError("shift must be between 1 and 7")

    if original_size == 0:
        return torch.tensor([], dtype=torch.uint8)


    if shift == 6:
        res = torch.zeros(original_size, dtype=torch.uint8)
        res_i = 0
        shift_i = 0
        while res_i < original_size:
            idx = shift_i // 8
            l_bits = shift_i - idx * 8
            r_bits = 8 - l_bits
            if l_bits == 0:
                res[res_i] = packed_data[idx]
            else:
                res[res_i] = ((packed_data[idx] << l_bits) & 0xFF) | (packed_data[idx + 1] >> r_bits)
            res_i += 1
            shift_i += 2
        return res



    packed_flat = packed_data.flatten()
    n = original_size

    if n == 1:
        # First element is stored in first 8 bits
        result = torch.zeros(1, dtype=torch.uint8)
        for bit_idx in range(8):
            byte_idx = bit_idx // 8
            bit_in_byte = bit_idx % 8
            bit_value = (packed_flat[byte_idx].item() >> bit_in_byte) & 1
            result[0] = result[0] | (bit_value << bit_idx)
        return result

    unpacked = torch.zeros(n, dtype=torch.uint8)
    bit_pos = 0

    # Read all 8 bits of first element
    for bit_idx in range(8):
        byte_idx = bit_pos // 8
        bit_in_byte = bit_pos % 8
        bit_value = (packed_flat[byte_idx].item() >> bit_in_byte) & 1
        unpacked[0] = unpacked[0] | (bit_value << bit_idx)
        bit_pos += 1

    # Read remaining elements
    for i in range(1, n):
        # Read lower (8-shift) bits
        for bit_idx in range(8 - shift):
            byte_idx = bit_pos // 8
            bit_in_byte = bit_pos % 8
            bit_value = (packed_flat[byte_idx].item() >> bit_in_byte) & 1
            unpacked[i] = unpacked[i] | (bit_value << bit_idx)
            bit_pos += 1

        # Upper shift bits come from lower shift bits of previous element
        prev_lower_bits = unpacked[i - 1].item() & ((1 << shift) - 1)
        unpacked[i] = unpacked[i] | (prev_lower_bits << (8 - shift))

    return unpacked


def trellis_encode_bit_shift(input_bits, scramble_bits=False, k=2):
    """
    Encodes input sequence of 8bit numbers using viterbi trellis encoding.
    One 8bit number have output edge to other if last 4 bits of current number
    are same as first 4 bits of next number. The output is a sequence of 8bit numbers this two times less elements.


    Args:
        input_bits (torch.Tensor): A tensor of shape (N) where N is the number of 8bit numbers.
        scramble_bits (bool): If True, scramble the input bits before encoding. Default is False.
    Returns:
        torch.Tensor: A tensor of shape (N / 2 + 1).
    """
    N = input_bits.size(0)
    N_OUT = N // 2 + 1

    # if scramble_bits:
    #     input_bits = ((input_bits >> 3) | (input_bits << 5))


    assert input_bits.dtype in [torch.uint8, torch.int8], "Input bits must be of type 8 bit type"
    if input_bits.dtype == torch.int8:
        input_bits = input_bits.to(torch.uint8)


    transitions = torch.zeros((256, 256), dtype=torch.float32)

    for i in range(256):
        shared_bits = i & (0b11111111 >> k)
        mask = (0b11111111 << k) & 0xFF
        for j in range(256):
            if shared_bits == (j & mask) >> k:
                transitions[i, j] = 1.0 / 2**(8 - k)
                # if i == 204:
                #     print(bin(i), bin(j), bin(shared_bits), bin(mask), transitions[i, j])

    states = torch.arange(256, dtype=torch.uint8)
    if scramble_bits:
        states = ((states >> 3) | (states << 5)) & 0xFF

        for i in range(256):
            if not i in states:
                raise ValueError("Scrambling resulted in non-unique states")

    states = states.unsqueeze(1).float()

    distance = (states - input_bits.unsqueeze(0).float())**2
    distance = torch.exp(-distance / (2 * (16.0 ** 2)))
    distance = distance / distance.sum(dim=0, keepdim=True)



    path = viterbi_path(
        prior=distance[:, 0],
        transmat=transitions,
        obslik=distance,
        scaled=True,
        ret_loglik=False
    )

    encoded_bits = pack(path.to(torch.uint8), shift=8 - k)

    return encoded_bits


def trellis_decode_bit_shift(encoded_bits, N, descramble_bits=False, k=2):
    """
    Decodes input sequence of 8bit numbers using viterbi trellis decoding.
    One 8bit number have output edge to other if last 4 bits of current number
    are same as first 4 bits of next number. The input is a sequence of 8bit numbers
    this two times less elements.


    Args:
        encoded_bits (torch.Tensor): A tensor of shape (N // 2 + 1) where N is the number of 8bit numbers.
        N (int): The number of output 8bit numbers after decoding.
        descramble_bits (bool): If True, descramble the output bits after decoding. Default is False.
    Returns:
        torch.Tensor: A tensor of shape (N).
    """
    res = unpack(encoded_bits, shift=8 - k, original_size=N)

    if descramble_bits:
        res = ((res >> 3) | (res << 5)) & 0xFF

    return res




def test_8bit():
    print("-"*40)
    scrambled_bits = False
    input_bits = torch.tensor([0b11001100, 0b11000011, 0b00110011, 0b00111100], dtype=torch.uint8)
    expected_encoded = torch.tensor([0b11001100, 0b00110011, 0b11000000], dtype=torch.uint8)

    encoded = trellis_encode(input_bits, scramble_bits=scrambled_bits)
    print("Encoded bits:", encoded)
    assert torch.equal(encoded, expected_encoded), "Encoded bits do not match expected output!"

    decoded = trellis_decode(encoded, N=input_bits.size(0), descramble_bits=scrambled_bits)
    print("Decoded bits:", decoded)
    assert torch.equal(input_bits, decoded), "Decoded bits do not match original input bits!"

    print("-"*40)
    print("\nTesting with random input bits:")
    input_bits = torch.tensor([90, 56, 32, 55, 128, 201, 253], dtype=torch.uint8)
    encoded = trellis_encode(input_bits, scramble_bits=scrambled_bits)
    print("Encoded bits:", encoded)

    decoded = trellis_decode(encoded, N=input_bits.size(0), descramble_bits=scrambled_bits)
    print("Input bits:", input_bits)
    print("Decoded bits:", decoded)
    print("Dispersion:", torch.std(input_bits.float() - decoded.float()).item())


    input_bits = torch.tensor([140,  92, 157, 124, 129, 152,  73, 148,  78, 170, 172, 112, 138,  62,
        106, 110,  64, 147,  86, 124,  98, 141,  44, 188, 176,  73, 134,  78,
         65,  90,  82, 169], dtype=torch.uint8)


    print("-"*40)
    print("\nTesting with real data:")
    encoded = trellis_encode(input_bits, scramble_bits=scrambled_bits)
    print("Encoded bits:", encoded)

    decoded = trellis_decode(encoded, N=input_bits.size(0), descramble_bits=scrambled_bits)
    print("Input bits:", input_bits)
    print("Decoded bits:", decoded)
    print("Dispersion:", torch.std(input_bits.float() - decoded.float()).item())


    scrambled_bits = True
    print("-"*40)
    print("\nTesting with real data and scrambling enabled:")
    encoded = trellis_encode(input_bits, scramble_bits=scrambled_bits)
    print("Encoded bits:", encoded)

    decoded = trellis_decode(encoded, N=input_bits.size(0), descramble_bits=scrambled_bits)
    print("Input bits:", input_bits)
    print("Decoded bits:", decoded)
    print("Dispersion:", torch.std(input_bits.float() - decoded.float()).item())


def test_4bits():
    scrambled_bits = False
    input_bits = torch.tensor([140,  92, 157, 124, 129, 152,  73, 148,  78, 170, 172, 112, 138,  62,
        106, 110,  64, 147,  86, 124,  98, 141,  44, 188, 176,  73, 134,  78,
         65,  90,  82, 169], dtype=torch.uint8) // 16


    print("-"*40)
    print("\nTesting with real data:")
    encoded = trellis_encode_4_bit(input_bits, scramble_bits=scrambled_bits)
    print("Encoded bits:", encoded)

    decoded = trellis_decode_4_bit(encoded, N=input_bits.size(0), descramble_bits=scrambled_bits)
    print("Input bits:", input_bits)
    print("Decoded bits:", decoded)
    print("Dispersion:", torch.std(input_bits.float() - decoded.float()).item())


    scrambled_bits = True
    print("-"*40)
    print("\nTesting with real data and scrambling enabled:")
    encoded = trellis_encode_4_bit(input_bits, scramble_bits=scrambled_bits)
    print("Encoded bits:", encoded)

    decoded = trellis_decode_4_bit(encoded, N=input_bits.size(0), descramble_bits=scrambled_bits)
    print("Input bits:", input_bits)
    print("Decoded bits:", decoded)
    print("Dispersion:", torch.std(input_bits.float() - decoded.float()).item())

    input_bits = torch.tensor([ 5, 10,  7, 10,  7, 10,  7,  4,  7,  7, 11,  8,  9,  8,  8,  7,  6,  9,
         6,  7, 10,  9, 11,  5,  6,  4,  9,  8, 10, 10,  9,  8],
       dtype=torch.int8)

    scrambled_bits = False
    print("-"*40)
    print("\nTesting with real data:")
    encoded = trellis_encode_4_bit(input_bits, scramble_bits=scrambled_bits)
    print("Encoded bits:", encoded)

    decoded = trellis_decode_4_bit(encoded, N=input_bits.size(0), descramble_bits=scrambled_bits)
    print("Input bits:", input_bits)
    print("Decoded bits:", decoded)
    print("Dispersion:", torch.std(input_bits.float() - decoded.float()).item())


    scrambled_bits = True
    print("-"*40)
    print("\nTesting with real data and scrambling enabled:")
    encoded = trellis_encode_4_bit(input_bits, scramble_bits=scrambled_bits)
    print("Encoded bits:", encoded)

    decoded = trellis_decode_4_bit(encoded, N=input_bits.size(0), descramble_bits=scrambled_bits)
    print("Input bits:", input_bits)
    print("Decoded bits:", decoded)
    print("Dispersion:", torch.std(input_bits.float() - decoded.float()).item())



def test_8bit_to_2bit():
    print("-"*40)
    scrambled_bits = False
    # input_bits = torch.tensor([0b11001100, 0b11000011, 0b00110011, 0b00111100], dtype=torch.uint8)
    # expected_encoded = torch.tensor([0b11001100, 0b00110011, 0b11000000], dtype=torch.uint8)

    # encoded = trellis_encode_bit_shift(input_bits, scramble_bits=scrambled_bits, k=4)
    # print("Encoded bits:", encoded)
    # assert torch.equal(encoded, expected_encoded), "Encoded bits do not match expected output!"

    # decoded = trellis_decode_bit_shift(encoded, N=input_bits.size(0), descramble_bits=scrambled_bits, k=4)
    # print("Decoded bits:", decoded)
    # assert torch.equal(input_bits, decoded), "Decoded bits do not match original input bits!"


    input_bits = torch.tensor([0b11101100,
                               0b10110011,
                               0b11001111,
                               0b00111100], dtype=torch.uint8)


# 11101100
#   10110011,
#     11001111,
#       00111100

#11101100111100


        # 11101100
        #   10110011,
        #     11001111,
        #       00111100

# 8 * k + k = 20

    expected_encoded = torch.tensor([0b11101100, 0b11110000], dtype=torch.uint8)
    encoded = trellis_encode_bit_shift(input_bits, scramble_bits=scrambled_bits, k=2)
    print("Encoded bits:", encoded)
    decoded = trellis_decode_bit_shift(encoded, N=input_bits.size(0), descramble_bits=scrambled_bits, k=2)
    print("Decoded bits:", decoded)
    assert torch.equal(input_bits, decoded), "Decoded bits do not match original input bits!"

    print("-"*40)
    print("\nTesting with random input bits:")
    input_bits = torch.tensor([90, 56, 32, 55, 128, 201, 253], dtype=torch.uint8)
    encoded = trellis_encode_bit_shift(input_bits, scramble_bits=scrambled_bits, k=2)
    print("Encoded bits:", encoded)

    decoded = trellis_decode_bit_shift(encoded, N=input_bits.size(0), descramble_bits=scrambled_bits, k=2)
    print("Input bits:", input_bits)
    print("Decoded bits:", decoded)
    print("Dispersion:", torch.std(input_bits.float() - decoded.float()).item())

    scrambled_bits = True
    print("-"*40)
    print("\nTesting with random input bits:")
    input_bits = torch.tensor([90, 56, 32, 55, 128, 201, 253], dtype=torch.uint8)
    encoded = trellis_encode_bit_shift(input_bits, scramble_bits=scrambled_bits, k=2)
    print("Encoded bits:", encoded)

    decoded = trellis_decode_bit_shift(encoded, N=input_bits.size(0), descramble_bits=scrambled_bits, k=2)
    print("Input bits:", input_bits)
    print("Decoded bits:", decoded)
    print("Dispersion:", torch.std(input_bits.float() - decoded.float()).item())
    
    
    scrambled_bits = False
    print("-"*40)
    print("\nTesting with random input bits:")
    input_bits = torch.tensor([95]*32, dtype=torch.uint8)
    encoded = trellis_encode_bit_shift(input_bits, scramble_bits=scrambled_bits, k=2)
    print("Encoded bits:", encoded)

    decoded = trellis_decode_bit_shift(encoded, N=input_bits.size(0), descramble_bits=scrambled_bits, k=2)
    print("Input bits:", input_bits)
    print("Decoded bits:", decoded)
    print("Dispersion:", torch.std(input_bits.float() - decoded.float()).item())


    scrambled_bits = True
    print("-"*40)
    print("\nTesting with random input bits:")
    input_bits = torch.tensor([95]*32, dtype=torch.uint8)
    encoded = trellis_encode_bit_shift(input_bits, scramble_bits=scrambled_bits, k=2)
    print("Encoded bits:", encoded)

    decoded = trellis_decode_bit_shift(encoded, N=input_bits.size(0), descramble_bits=scrambled_bits, k=2)
    print("Input bits:", input_bits)
    print("Decoded bits:", decoded)
    print("Dispersion:", torch.std(input_bits.float() - decoded.float()).item())

if __name__ == "__main__":
    #test_8bit()
    #test_4bits()
    test_8bit_to_2bit()
