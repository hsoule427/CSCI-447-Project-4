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

# Custom imports
import shared_functions as sf

# ----------------------------------------------
class FFNN():
    ''' ----------------------------------------------
    Constructor

    layer_sizes    Contains the number of neurons
                   in the i-th layer

    db_type        A string that is either
                   'classification' or 'regression'
    '''

    def __init__(self, layer_sizes, data, db_type, learning_rate,
                 num_epochs=100):

        # Initialization from constructor parameters
        self.layer_sizes = layer_sizes
        self.data = data
        self.db_type = db_type
        self.old_data = self.data[:]
        self.learning_rate = learning_rate

        self.epochs = [[] for x in range(num_epochs)]

        # Initializes weights via a normal distribution.
        self.weight_vec = [np.random.randn(y, x) / np.sqrt(x)
                           for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

        # Initializes biases via a normal distribution.
        self.bias_vec = [np.random.randn(x, 1) for x in self.layer_sizes[1:]]

        # Start the learning process...
        self.grad_desc()

    @classmethod
    def init_no_weights(self, data, learning_rate, layer_sizes=None):
        if layer_sizes:
            return FFNN(layer_sizes, data, None, learning_rate)
        return FFNN(None, data, None, learning_rate)

    ''' ----------------------------------------------
    Returns the output layer produced from in_act_vec
    
    in_act_vec    An activation vector of some layer
    '''

    def feed_forward(self, in_act_vec):
        for b, w in zip(self.bias_vec, self.weight_vec):
            in_act_vec = sf.sigmoid(np.dot(w, in_act_vec) + b)
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
            batches = [self.data[x: x + len_batch]
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
            if self.db_type == 'classification':
                if print_partial_progress is False:
                    if e == 0 or e == num_epoch - 1:
                        num_correct, total = self.zero_one_loss()
                        print('Epoch {}: {} / {}'.format(e, num_correct, total))
                else:
                    num_correct, total = self.zero_one_loss()
                    print('Epoch {}: {} / {}'.format(e, num_correct, total))

            elif self.db_type == 'regression':
                if print_partial_progress is False:
                    if e == 0 or e == num_epoch - 1:
                        avg_error = self.regression_error()
                        print("Epoch {}'s average error: {}".format(e, avg_error))

                else:
                    avg_error = self.regression_error()
                    print("Epoch {}'s average error: {}".format(e, avg_error))

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
        # find every layer's pre-and-post-sigmoid-activation vector
        for b, w in zip(self.bias_vec, self.weight_vec):
            z = np.dot(w, a) + b
            z_vecs.append(z)
            a = sf.sigmoid(z)
            a_vecs.append(a)

        # Notice this is the same as the "for layer_idx..." loop below.
        # We need to do this first step at the last layer in
        # a particular way, so it goes outside of the loop
        delta_layer = self.cost_prime(a_vecs[-1], desired_out) * sf.sigmoid_prime(z_vecs[-1])
        delta_b[-1] = delta_layer
        delta_w[-1] = np.dot(delta_layer, a_vecs[-2].transpose())
        for layer in range(2, len(self.layer_sizes)):
            delta_layer = np.dot(self.weight_vec[-layer + 1].transpose(), delta_layer)\
                          * sf.sigmoid_prime(z_vecs[-layer])
            delta_b[-layer] = delta_layer
            delta_w[-layer] = np.dot(delta_layer, a_vecs[-layer - 1].transpose())

        return delta_b, delta_w

    
    '''
    @brief          compute the overall fitness of the model with one pass thru all the data
    @param fit_fxn  a reference to the fitness function we will use, different for classification and regression
    '''
    def get_fitness(self, fit_fxn):
        total = 0
        # Loop over all points
        for a, desired_out in self.data:
            in_act_vec = a
            a_vecs = [in_act_vec]
            z_vecs = []
            
            # For every weight vector and respective layer bias,
            # find every layer's pre-and-post-sigmoid-activation vector
            for b, w in zip(self.bias_vec, self.weight_vec):
                z = np.dot(w, a) + b
                z_vecs.append(z)
                a = sf.sigmoid(z)
                a_vecs.append(a)
            
            total += fit_fxn(a_vecs[-1], desired_out)
        return total / len(self.data)

    def cost(self, out_acts, desired_out):
        return(np.sum((out_acts-desired_out) ** 2))


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

        return num_correct, total

    def set_weight(self, weight):
        self.weight_vec = weight
    
    def set_biases(self, biases):
        self.bias_vec = biases

    def regression_error(self):
        avg_error = 0

        for actual_out, desired_out in self.old_data:
            avg_error += (self.feed_forward(actual_out) - desired_out)**2

        return avg_error / len(self.old_data)