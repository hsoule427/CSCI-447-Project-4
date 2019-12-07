import numpy as np
from FFNN import FFNN
import shared_functions as sf
from copy import deepcopy

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
    c1 = np.random.rand() * 1.5
    c2 = np.random.rand() * 1.5
    return 0.1 * v_prev + c1 * (pbest - p) + c2 * (gbest - p)



'''
@param db               the db object
@param layer_sizes      the number of neurons in each layer our neural net
@param learning_rate    user defined learning rate (tune this?)
@param hp               a set of the tunable hyperparameters, 
                        in this case an array of three params: [c1, c2, inertia]
'''
def main_loop(db, layer_sizes, learning_rate, hp, epochs=100):
    # Initialize set of particles/weights (1d)
    particles = [np.array([np.random.random() for i in range(sf.calc_total_vec_length(layer_sizes))]) \
                    for i in range(20)]
    
    p_best_pos = deepcopy(particles) # Stores list of best positions
    p_best_scores = [float('inf') for i in range(len(particles))] # Stores list of best scores
    
    g_best_pos = deepcopy(particles[0])
    g_best_score = float('inf')
    # Initialize the set of velocities, one for each particle
    v = [np.zeros(len(particles[0])) for i in range(len(particles))]
    # Initialize our net
    ffnn = FFNN(db.get_data(), learning_rate)
    # Set the hyperparameters
    c1 = hp[0]
    c2 = hp[1]
    inertia = hp[2]
    
    for e in range(epochs):
        print('EPOCH: ', e)
        for i,p in enumerate(particles):
            weight_vec, bias_vec = sf.encode_weight_and_bias(p, layer_sizes)
            ffnn.set_weight(weight_vec)
            ffnn.set_biases(bias_vec)
            
            cost = ffnn.get_fitness(sf.classification_error)
            
            if cost < g_best_score:
                g_best_score = cost
                g_best_pos = deepcopy(p)
            
            if cost < p_best_scores[i]:
                p_best_scores[i] = cost
                p_best_pos[i] = deepcopy(p)
            
            # calculate velocity
            v[i] = velocity(v[i], p_best_pos[i], g_best_pos[i], p, c1, c2, inertia)
            particles[i] = particles[i] + v[i]
        
        weight_vec, bias_vec = sf.encode_weight_and_bias(g_best_pos, layer_sizes)
        ffnn.set_weight(weight_vec)
        ffnn.set_biases(bias_vec)
        fitness = ffnn.get_fitness(sf.classification_error)
        print("CURRENT BEST FITNESS = ", fitness)
        print("AVG DISTANCE = ", sf.calc_avg_distance(particles))
    
    # Get final fitness and avg distance, and return
    weight_vec, bias_vec = sf.encode_weight_and_bias(g_best_pos, layer_sizes)
    ffnn.set_weight(weight_vec)
    ffnn.set_biases(bias_vec)
    fitness = ffnn.get_fitness(sf.classification_error)
    avg_distance = sf.calc_avg_distance(particles)
    return fitness, avg_distance

            
def test_velocity(layer_sizes):
    p1 = np.array([np.random.random() for i in range(5)])
    p2 = np.array([np.random.random() for i in range(5)])
    p3 = np.array([np.random.random() for i in range(5)])
    print("P1:")
    print(p1)
    print("P2:")
    print(p2)
    print("P3:")
    print(p3)
    v_prev = 0
    v = velocity(v_prev, p1, p2, p3)
    print("Velocity:")
    print(v)

    







