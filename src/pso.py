import numpy as np
from FFNN import FFNN
import shared_functions as sf
from copy import deepcopy

def velocity(v_prev, p_best, g_best, p):
    u1 = np.random.rand() * 1.6
    u2 = np.random.rand() * 1.6
    # TODO inertia weight
    return v_prev + u1 * (p_best - p) + u2 * (g_best - p)





def main_loop(db, layer_sizes, learning_rate, epochs=100):
    # Initialize set of particles/weights (1d)
    particles = [np.random.randn(sf.calc_total_vec_length(layer_sizes)) for i in range(20)]
    p_best_pos = deepcopy(particles) # Stores list of best positions
    p_best_scores = [float('inf') for i in range(len(particles))] # Stores list of best scores
    g_best_pos = deepcopy(particles[0])
    g_best_score = float('inf')
    v = [np.zeros(len(particles[0])) for i in range(len(particles))]
    ffnn = FFNN(db.get_data(), learning_rate)
    for e in range(epochs):
        for i,p in enumerate(particles[0:1]):
            weight_vec, bias_vec = sf.encode_weight_and_bias(p, layer_sizes)
            ffnn.set_weight(weight_vec)
            ffnn.set_biases(bias_vec)
            cost = ffnn.get_fitness()
            if cost < g_best_score:
                g_best_score = cost
                g_best_pos = deepcopy(p)
            if cost < p_best_scores[i]:
                p_best_scores[i] = cost
                p_best_pos[i] = deepcopy(p)
            # caluclate velocity
            v[i] = velocity(v[i], p_best_pos[i], g_best_pos[i], p)
            particles[i] = particles[i] + v[i]
        
        weight_vec, bias_vec = sf.encode_weight_and_bias(g_best_pos, layer_sizes)
        ffnn.set_weight(weight_vec)
        ffnn.set_biases(bias_vec)
        cost = ffnn.get_fitness()
        print("COST = ", cost)

            






