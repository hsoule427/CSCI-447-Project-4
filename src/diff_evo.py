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


def mutate(x1,x2,x3,beta):
    return x1 + beta * (x2 - x3)


def calculate_fitness(ffnn, layer_sizes, weight_vec, db_type):
    weights, biases = sf.encode_weight_and_bias(weight_vec, layer_sizes)
    ffnn.set_weight(weights)
    ffnn.set_biases(biases)
    if db_type == 'classification':
        # return ffnn.get_fitness(sf.classification_error)
        return ffnn.zero_one_loss()[0]
    else: 
        return ffnn.regression_error()[0]


def get_best_fitness(ffnn, layer_sizes, population, db_type):
    best = 0 if db_type == 'classification' else float('inf')
    best_idx = 0
    for i,p in enumerate(population):
        weights, biases = sf.encode_weight_and_bias(p, layer_sizes)
        ffnn.set_weight(weights)
        ffnn.set_biases(biases)
        if db_type == 'classification':
            fitness = ffnn.zero_one_loss()[0]
            if fitness > best:
                best = fitness
                best_idx = i
        else:
            fitness = ffnn.regression_error()[0]
            if fitness < best:
                best = fitness
                best_idx = i
    
    return best, population[best_idx]
    

def binomial_crossover(x,v,pr):
    u = deepcopy(v)
    for i in range(len(u)):
        r = np.random.random()
        if r <= pr:
            u[i] = x[i]
    return u



def main_loop(data, db_type, layer_sizes, learning_rate, hp, generations=100):
    beta = hp[0]
    pr = hp[1]
    
    ffnn = FFNN.init_no_weights(data, learning_rate)
    population = [np.array([np.random.random() for i in range(sf.calc_total_vec_length(layer_sizes))]) \
                    for i in range(10)]
    
    for g in range(generations):
        print("GENERATION: ", g)
        for i,p in enumerate(population):
            rand_idxs = get_rand_indices(i, len(population))
            
            x1 = population[rand_idxs[0]]
            x2 = population[rand_idxs[1]]
            x3 = population[rand_idxs[2]]
            
            # Find candidate vector
            v = mutate(x1,x2,x3,beta)
            u = binomial_crossover(p,v,pr)
            
            # Calculate fitness with candidate vector
            fitness_u = calculate_fitness(ffnn, layer_sizes, v, db_type)
            print('FITNESS U: ', fitness_u)
            
            # Now do same for current vector in population
            fitness_p = calculate_fitness(ffnn, layer_sizes, p, db_type)
            print('FITNESS P: ', fitness_p)
            
            if db_type == 'classification' and fitness_u > fitness_p:
                population[i] = u
                print('ADDING U')
                
            
            elif db_type == 'regression' and fitness_u < fitness_p:
                population[i] = u
                print('ADDING U')
            print('---------------------------------------')
        
        print('AVG DISTANCE: ', sf.calc_avg_distance(population))
        # if db_type == 'classification':
        #     fitness = ffnn.zero_one_loss()[0]
        # else:
        #     fitness = ffnn.regression_error()
        fitness, pos = get_best_fitness(ffnn, layer_sizes, population, db_type)
        print('CURRENT BEST FITNESS: ', fitness)

    # final_fitness = calculate_fitness(ffnn, layer_sizes, population[0])
    final_fitness, pos = get_best_fitness(ffnn, layer_sizes, population, db_type)
    avg_dist = sf.calc_avg_distance(population)
    return final_fitness, avg_dist
            



            
            


