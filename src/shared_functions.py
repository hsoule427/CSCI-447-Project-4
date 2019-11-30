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