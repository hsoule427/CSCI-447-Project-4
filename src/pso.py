import numpy as np
from FFNN import FFNN
import shared_functions as sf
from copy import deepcopy
import sys

'''
@param v_prev   the previous velocity
@param p_best   particle p's best position
@param g_best   the global best position thus far
@param p        the current postion of this particle
@c1             acceleration coefficient for pbest 
@c2             acceleration coefficient for gbest
@inertia        the inertia
'''
def velocity(v_prev, pbest, gbest, p, c1, c2, inertia):
    return inertia * v_prev + c1 * (pbest - p) + c2 * (gbest - p)



'''
@param data             the dataset
@param layer_sizes      the number of neurons in each layer our neural net
@param learning_rate    user defined learning rate (tune this?)
@param hp               a set of the tunable hyperparameters, 
                        in this case an array of three params: [c1, c2, inertia]
'''
def main_loop(data, db_type, layer_sizes, learning_rate, hp, epochs=100):
    # Initialize set of particles/weights (1d)
    particles = [np.array([np.random.random() for i in range(sf.calc_total_vec_length(layer_sizes))]) \
                    for i in range(5)]
    
    p_best_pos = deepcopy(particles)

    print('INITIAL PARTICLE SET:')
    for p in particles:
        print(p[0:5])
    
    if db_type == 'classification':
        p_best_scores = [0 for i in range(len(particles))]
    else: # regression
        p_best_scores = [float(sys.maxsize) for i in range(len(particles))]
    
    g_best_pos = deepcopy(particles[0])
    g_best_score = 0 if db_type == 'classification' else float(sys.maxsize)
    # Initialize the set of velocities, one for each particle
    v = [np.zeros(len(particles[0])) for i in range(len(particles))]
    # Initialize our net
    ffnn = FFNN.init_no_weights(data, learning_rate, layer_sizes=layer_sizes)
    # Set the hyperparameters
    c1 = hp[0]
    c2 = hp[1]
    inertia = hp[2]
    start_fts = 0

    for e in range(epochs):
        print('EPOCH: ', e)
        for i,p in enumerate(particles):
            print('PARTICLE ', i, ': ')
            print(p[0:5])
            weight_vec, bias_vec = sf.encode_weight_and_bias(p, layer_sizes)
            ffnn.set_weight(weight_vec)
            ffnn.set_biases(bias_vec)
            print('CURRENT GBEST: ', g_best_score)
            print('CURRENT PBEST: ', p_best_scores[i])
            
            if db_type == 'classification':
                cost = ffnn.zero_one_loss()[0]
                print('CURRENT FITNESS: ', cost)
                if cost > g_best_score:
                    print('UPDATING GBEST')
                    g_best_score = cost
                    g_best_pos = deepcopy(p)                
                if cost > p_best_scores[i]:
                    print('UPDATING PBEST')
                    p_best_scores[i] = cost
                    p_best_pos[i] = deepcopy(p)

            else: # regression dataset
                cost = ffnn.regression_error()[0]
                print('CURRENT FITNESS: ', cost)
                if cost < g_best_score:
                    print('UPDATING GBEST')
                    g_best_score = cost
                    g_best_pos = deepcopy(p)
                if cost < p_best_scores[i]:
                    print('UPDATING PBEST')
                    p_best_scores[i] = cost
                    p_best_pos[i] = deepcopy(p)

            # calculate velocity
            v[i] = velocity(v[i], p_best_pos[i], g_best_pos, p, c1, c2, inertia)
            print('COMPUTED VELOCITY: ')
            print(v[i][0:5])
            particles[i] = particles[i] + v[i]
            print('UPDATED PARTICLE:')
            print(particles[i][0:5])
            
            print('\n\n')
        if e == 0:
            start_fts = g_best_score
        
        weight_vec, bias_vec = sf.encode_weight_and_bias(g_best_pos, layer_sizes)
        ffnn.set_weight(weight_vec)
        ffnn.set_biases(bias_vec)
        
        if db_type == 'classification':
            fitness = ffnn.zero_one_loss()[0]
        else: # regression
            fitness = ffnn.regression_error()[0]
    
    print('UPDATED PARTICLES:')
    for p in particles:
        print(p[0:5])
    
    # Get final fitness and avg distance, and return
    weight_vec, bias_vec = sf.encode_weight_and_bias(g_best_pos, layer_sizes)
    ffnn.set_weight(weight_vec)
    ffnn.set_biases(bias_vec)
    if db_type == 'classification':
        fitness = ffnn.zero_one_loss()[0]
    else:
        fitness = ffnn.regression_error()[0]
    avg_distance = sf.calc_avg_distance(particles)
    return start_fts, fitness, g_best_pos

    







