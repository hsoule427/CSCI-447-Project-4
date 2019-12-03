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
@brief  encode the multi-dimensional set of weights from ffnn into a 1D array
        to be used with the different EAs
'''
def encode_weights(weight_vec):
    encoded_weights = []
    # Yes it's three embedded for loops don't judge me
    for array in weight_vec:
        for vec in array:
            for weight in vec:
                encoded_weights.append(weight)
    
    return encoded_weights

'''----------------------------------------------
@brief  decode the 1d encoding back to original form for the ffnn
'''
def decode_weights(weight_vec, layer_sizes):
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
@brief  Calculate the length of a 1d array of weights from the layer sizes
'''
def calc_weight_vec_length(layer_sizes):
    return sum([layer_sizes[i-1] * layer_sizes[i] for i in range(1,len(layer_sizes))])
            
    

        

