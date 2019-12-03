''' ----------------------------------------------
@file:    FFNN.py
@authors: Dana Parker, Henry Soule, Troy Oster, George Engel
'''

# ----------------------------------------------
# Standard-library imports

# Third-party imports
import numpy as np

''' ----------------------------------------------
The sigmoid activation function

w_dot_a_plus_b   A weighted sum which equals
                 the dot product of the weight vector (w)
                 and activation vector (a)
                 plus the bias vector (b)
'''
def sigmoid(w_dot_a_plus_b):
    return 1.0 / (1.0 + np.exp(-w_dot_a_plus_b))

''' ----------------------------------------------
The derivative of the sigmoid function

w_dot_a_plus_b  See sigmoid()
'''
def sigmoid_prime(w_dot_a_plus_b):
    return sigmoid(w_dot_a_plus_b) * (1 - sigmoid(w_dot_a_plus_b))

'''----------------------------------------------
@brief  decode the multi-dimensional set of weights from ffnn into a 1D array
        to be used with the different EAs
'''
def decode_weights(weight_vec):
    encoded_weights = []
    # Yes it's three embedded for loops don't judge me
    for array in weight_vec:
        for vec in array:
            for weight in vec:
                encoded_weights.append(weight)
    
    return encoded_weights

'''----------------------------------------------
@brief  encode the 1d weight vec to original form for the ffnn
'''
def encode_weights(weight_vec, layer_sizes):
    new_weight_vec = []
    
    idx = 0
    for i in range(1, len(layer_sizes)):
        new_vec = []
        array_len = layer_sizes[i] # Length of outer array
        vec_len = layer_sizes[i - 1] # Length of each inner array
        for j in range(array_len):
            new_vec.append(weight_vec[idx:idx+vec_len])
            idx += vec_len
        new_weight_vec.append(np.array(new_vec))
    
    return new_weight_vec

'''---------------------------------------------
@brief  Decode bias vec to 1d array
'''
def decode_biases(biases):
    vec = []
    for arr in biases:
        for num in arr:
            vec.append(num[0])
    return vec

'''----------------------------------------------
@brief   
'''
def encode_biases(biases, layer_sizes):
    vec = []
    idx = 0
    for l in layer_sizes[1:]:
        # vec.append(np.array(biases[idx:idx+l]))
        arr = []
        for i in range(idx, idx+l):
            arr.append([biases[i]])
        vec.append(np.array(arr))
        idx += l
    
    # for arr in vec:
    #     for i,num in enumerate(arr):
    #         arr[i] = [num]

    return vec


'''---------------------------------------------
@brief  Calculate the length of a 1d array of weights from the layer sizes
'''
def calc_weight_vec_length(layer_sizes):
    return sum([layer_sizes[i-1] * layer_sizes[i] for i in range(1,len(layer_sizes))])
            
    

        

