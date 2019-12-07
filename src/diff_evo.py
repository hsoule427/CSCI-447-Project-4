import numpy as np
from FFNN import FFNN
import shared_functions as sf
from copy import deepcopy

'''
@brief          Generate 3 random indices from population
@param idx      The index of the current pop member we are using (so skip over)
@param pop_size Size of the population (high end of range to generate ints within)
'''
def get_rand_indices(idx, pop_size):
    indices = []
    while(len(indices) < 3):
        num = np.random.randint(0,pop_size)
        if num not in indices and num != idx:
            indices.append(num)
    return indices


def candidate_vector(x1,x2,x3,beta):
    return x1 + beta * (x2 - x3)


def calculate_fitness(ffnn, layer_sizes, weight_vec, db_type):
    weights, biases = sf.encode_weight_and_bias(weight_vec, layer_sizes)
    ffnn.set_weight(weights)
    ffnn.set_biases(biases)
    if db_type == 'classification':
        return ffnn.get_fitness(sf.classification_error)
    else: 
        return ffnn.get_fitness(sf.squared_error)


def binomial_crossover(x,v,pr=0.7):
    u = deepcopy(v)
    for i in range(len(u)):
        r = np.random.random()
        if r <= pr:
            u[i] = x[i]
    return u



def main_loop(db, layer_sizes, learning_rate, generations=100):
    beta = 0.08
    ffnn = FFNN(db.get_data(), learning_rate)
    population = [np.random.randn(sf.calc_total_vec_length(layer_sizes)) for i in range(20)]
    for g in range(generations):
        print("GENERATION: ", g)
        for i,p in enumerate(population):
            rand_idxs = get_rand_indices(i, len(population))
            x1 = population[rand_idxs[0]]
            x2 = population[rand_idxs[1]]
            x3 = population[rand_idxs[2]]
            v = candidate_vector(x1,x2,x3,beta)
            # Calculate fitness with candidate vector
            fitness_v = calculate_fitness(ffnn, layer_sizes, v, db.get_dataset_type())
            # Now do same for current vector in population
            fitness_p = calculate_fitness(ffnn, layer_sizes, p, db.get_dataset_type())
            if fitness_v < fitness_p:
                population[i] = binomial_crossover(p,v)
        print('AVG DISTANCE: ', sf.calc_avg_distance(population))
    print('FINAL WEIGHTS:')
    print(population)

    final_fitness = calculate_fitness(ffnn, layer_sizes, population[0])
    print('FINAL FITNESS: ', final_fitness)

            



            
            


