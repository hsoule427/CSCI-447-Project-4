''' ----------------------------------------------
@file:    FFNN.py
@authors: Dana Parker, Henry Soule, Troy Oster, George Engel
'''

# ----------------------------------------------
# Standard-library imports
import random
import math
import sys

# Third-party imports
import numpy as np

# ----------------------------------------------
class FFNN():

    ''' ----------------------------------------------
    Constructor

    layer_sizes    Contains the number of neurons
                   in the i-th layer

    db_type        A string that is either
                   'classification' or 'regression'
    '''
    def __init__(self, layer_sizes, db_type,
        data, learning_rate,
        class_list=None, num_epochs=200):

        # Initialization from constructor parameters
        self.layer_sizes = layer_sizes
        self.db_type = db_type

        if db_type == 'classification':
            self.act_fn = sigmoid
            self.act_fn_prime = sigmoid_prime
        elif db_type == 'regression':
            self.act_fn = linear_act_fn
            self.act_fn_prime = linear_act_fn
        else:
            print('Invalid database type. Quitting.')
            sys.exit()

        self.data = data
        self.old_data = self.data[:]
        
        self.learning_rate = learning_rate
        
        if class_list:
            self.class_list = class_list

        self.epochs = [[] for x in range(num_epochs)]

        # Initializes weights via a normal distribution.
        self.weight_vec = [np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        
        # print("WEIGHT VECTOR:")
        # print(self.weight_vec)

        # Initializes biases via a normal distribution.
        self.bias_vec = [np.random.randn(x, 1) for x in self.layer_sizes[1:]]

        # Start the learning process...
        self.grad_desc()

    ''' ----------------------------------------------
    Returns the output layer produced from in_act_vec
    
    in_act_vec    An activation vector of some layer
    '''
    def feed_forward(self, in_act_vec):
        for b, w in zip(self.bias_vec, self.weight_vec):
            in_act_vec = (self.act_fn)(np.dot(w, in_act_vec) + b)
        return in_act_vec

    ''' ----------------------------------------------
    Trains the neural network via stochastic gradient descent
    with mini-batches.

    print_partial_progress  A boolean for whether we want to
                            print the evaluation of EVERY epoch
                            (WARNING: slow if True)
    '''
    def grad_desc(self, print_partial_progress=False):

        num_data = len(self.data)
        num_epoch = len(self.epochs)
        len_batch = math.ceil(num_data / 10)

        # The gradient descent itself for every epoch
        for e in range(num_epoch):

            # Randomly shuffle the training data
            random.shuffle(self.data)

            # Split the data into mini-batches
            batches = [self.data[x : x + len_batch]
                for x in range(0, num_data, len_batch)]

            # For every mini-batch,
            # update the entire networks's weights and biases
            # via one gradient descent iteration
            # (this step uses the back propagation)
            for curr_batch in batches:
                new_b = [np.zeros(b.shape) for b in self.bias_vec]
                new_w = [np.zeros(w.shape) for w in self.weight_vec]
                # Perform backpropagation and apply changes
                for ex, desired_out in curr_batch:
                    # print("EX: ", ex)
                    # print("DESIRED OUT: ", desired_out)
                    # print("-------------------")
                    delta_b, delta_w = self.back_prop(ex, desired_out)
                    new_b = [bias + change for bias, change in zip(new_b, delta_b)]
                    new_w = [weight + change for weight, change in zip(new_w, delta_w)]

                # Apply momentum
                self.weight_vec = \
                    [w - (self.learning_rate / len_batch) * nw
                    for w, nw
                    in zip(self.weight_vec, new_w)]

                self.bias_vec = \
                    [b - (self.learning_rate / len_batch) * nb
                    for b, nb
                    in zip(self.bias_vec, new_b)]

            # Print results of the epochs
            # print_partial_progress is set manually
            # in the function parameters
            if print_partial_progress is False:
                if e == 0 or e == num_epoch - 1:
                    num_correct, total = self.zero_one_loss()
                    print('Epoch {}: {} / {}'.format(e, num_correct, total))
            else:
                num_correct, total = self.zero_one_loss()
                print('Epoch {}: {} / {}'.format(e, num_correct, total))

    '''
    ----------------------------------------------
    Basically finding the partial derivatives of
    the cost with respect to both the weights and the biases
    (I say 'basically', but I hardly understand it, and I wrote it)
    '''
    def back_prop(self, in_act_vec, desired_out):

        # Variable declarations
        delta_b = [np.zeros(b.shape) for b in self.bias_vec]
        delta_w = [np.zeros(w.shape) for w in self.weight_vec]

        a = in_act_vec
        a_vecs = [in_act_vec]
        z_vecs = []

        # For every weight vector and respective layer bias,
        # find every layer's pre-and-post-actfn-activation vector
        for b, w in zip(self.bias_vec, self.weight_vec):

            z = np.dot(w, a) + b
            z_vecs.append(z)
            a = (self.act_fn)(z)
            a_vecs.append(a)

        # Notice this is the same as the "for layer_idx..." loop below.
        # We need to do this first step at the last layer in
        # a particular way, so it goes outside of the loop
        delta_l = self.cost_prime(a_vecs[-1], desired_out) * (self.act_fn_prime)(z_vecs[-1])
        delta_b[-1] = delta_l
        delta_w[-1] = np.dot(delta_l, a_vecs[-2].transpose())
        for L in range(2, len(self.layer_sizes)):

            z = z_vecs[-L]
            sp = (self.act_fn_prime)(z)
            delta_l = np.dot(self.weight_vec[-L+1].transpose(), delta_l) * sp
            delta_b[-L] = delta_l
            delta_w[-L] = np.dot(delta_l, a_vecs[-L-1].transpose())

        return (delta_b, delta_w)

    ''' ----------------------------------------------
    The derivative of our cost function
    if the cost function is (a - y)^2
    (I drop the constant 2)
    
    out_acts        A vector of activations of some layer
    desired_out     The desired outputs
    '''
    def cost_prime(self, out_acts, desired_out):
        return out_acts - desired_out
    
    ''' ----------------------------------------------
    Returns the percentage of correct classifications
    '''
    def zero_one_loss(self):

        num_correct = 0
        total = len(self.old_data)

        for actual_out, desired_out in self.old_data:

            # If the index of the highest activation
            # is equal to the index of the desired output,
            # then we classified correctly
            if np.argmax((self.feed_forward(actual_out))) == np.argmax(desired_out):
                num_correct += 1
        
        return (num_correct, total)

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

''' ----------------------------------------------
The activation function used for regression
It doesn't do anything, but we need a function to
make activation functions generic
'''
def linear_act_fn(self, x):
    return x
