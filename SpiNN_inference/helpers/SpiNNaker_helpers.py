""""
Utility functions to create convolutional connectivity in a neural network
by mapping input neuron indices to output neuron indices based on kernel geometry.

The input layer is padded to ensure the kernel can center on edge neurons.
A 'segment' is a local neighborhood of input neurons that falls under the kernel
at a specific spatial location. Each segment corresponds to a single post-synaptic neuron.
Treating the input matrix as a grid of neuron ids, we extract segments
to create (source_id, target_id, weight) tuples.
"""

import math
import numpy as np

def extract_segments(x, kernel_size, stride=(1,1), padding=0):
    Ky, Kx = kernel_size
    Sy, Sx = stride
    H, W = x.shape

    # Output size
    out_y = (H - Ky) // Sy + 1
    out_x = (W - Kx) // Sx + 1

    segments = []

    for iy in range(0, out_y):
        for ix in range(0, out_x):
            y = iy * Sy
            x_ = ix * Sx
            segment = x[y:y+Ky, x_:x_+Kx]
            segments.append(segment.flatten())

    return np.array(segments)

def same_padded_input_size(input_size, kernel_size, stride=1):
    print(input_size)
    H_in, W_in = input_size
    K_h, K_w = kernel_size
    
    if isinstance(stride, int):
        S_h = S_w = stride
    else:
        S_h, S_w = stride

    H_out = math.ceil(H_in / S_h)
    W_out = math.ceil(W_in / S_w)

    P_h_total = max((H_out - 1) * S_h + K_h - H_in, 0)
    P_w_total = max((W_out - 1) * S_w + K_w - W_in, 0)

    pad_top = P_h_total // 2
    pad_bottom = P_h_total - pad_top
    pad_left = P_w_total // 2
    pad_right = P_w_total - pad_left

    H_padded = H_in + pad_top + pad_bottom
    W_padded = W_in + pad_left + pad_right

    return (H_padded, W_padded), ((pad_top, pad_bottom), (pad_left, pad_right))


def extract_segments(Input_matrix, padded_size, kernel_size, stride=(1,1), padding=None):
    Ky, Kx = kernel_size
    Sy, Sx = stride

    # Apply padding
    padded_I = np.ones(padded_size, dtype=Input_matrix.dtype)*-1
    if padding is not None:
        pad_top, pad_bottom = padding[0]
        pad_left, pad_right = padding[1]
        padded_I[pad_top:pad_top+Input_matrix.shape[0], pad_left:pad_left+Input_matrix.shape[1]] = Input_matrix

    H, W = padded_size
    out_y = (H - Ky) // Sy + 1
    out_x = (W - Kx) // Sx + 1

    segments = []
    for iy in range(out_y):
        for ix in range(out_x):
            y = iy * Sy
            x = ix * Sx
            segment = padded_I[y:y+Ky, x:x+Kx]
            segments.append(segment)  # flatten if you want 1D

    return np.array(segments), (out_y, out_x)

def conv_connection(input_size, kernel, stride = (1, 1), padding_value = -1):
    kernel_size = kernel.shape
    padded_size, padding_same_size = same_padded_input_size(input_size, kernel_size, stride)
    input_dummy = np.arange(0, np.prod(input_size)).reshape(input_size)
    segments, output_size = extract_segments(
        input_dummy, 
        padded_size,
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding_same_size
    )
    assert(output_size == (int(np.sqrt(len(segments))), int(np.sqrt(len(segments)))))
    connection_list = []
    for n_id_out, seg in enumerate(segments):
        for (n_id_in, w) in (zip(seg.flatten(), kernel.flatten())):
            if n_id_in != padding_value:
                connection_list.append((n_id_in, n_id_out, w, 1))

    return connection_list, output_size