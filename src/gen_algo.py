# https://towardsdatascience.com/continuous-genetic-algorithm-from-scratch-with-python-ff29deedd099

import numpy as np
from numpy.random import randint
import random
from FFNN import FFNN
import random
import shared_functions as sf

# def init_example(num_genes,
#                  lower = -4, upper = 4):
#     example = [round(rnd() * (upper - lower) + lower, 1)
#                 for i in range(num_genes)]
#     return example

def init_population(num_genes, num_chromosomes):
    return [np.array([np.random.random() for i in range(num_genes)]) \
                    for i in range(num_chromosomes)]

def tournament_selection(pop, ffnn, fit_fcn, print_selection,
                       k=2):
    selections = []
    idx = []

    if print_selection is True:
        print('begin tournament selection...')

    # Select the (non-duplicate) individuals randomly
    while len(selections) != k:
        rand_idx = random.randint(0, len(pop) - 1)
        if rand_idx not in idx:
            idx.append(rand_idx)
            selections.append(pop[rand_idx])

    if print_selection is True:
        print('1st candidate:', selections[0][0], selections[0][1], selections[0][2], selections[0][3], selections[0][4])
        print('2nd candidate:', selections[1][0], selections[1][1], selections[1][2], selections[1][3], selections[1][4])

    # Evaluate the fitness of the selections
    fitnesses = []
    for curr_chrom in selections:
        w_vec, b_vec = sf.encode_weight_and_bias(curr_chrom, ffnn.layer_sizes)
        ffnn.set_weight(w_vec)
        ffnn.set_biases(b_vec)
        fitnesses.append(ffnn.get_fitness(fit_fcn))

    # Return the individual with the lower (better) fitness
    if print_selection is True:
        temp = selections[fitnesses.index(min(fitnesses))]
        print('selected candidate:', temp[0], temp[1], temp[2], temp[3], temp[4])
    return selections[fitnesses.index(min(fitnesses))]

def roulette_wheel_selection(cumulative_sum, prob):
    rv = list(cumulative_sum)
    rv.append(prob)
    rv = sorted(rv)
    return rv.index(prob)


def crossover(pop, fitnesses, print_crossover):

    # Get the normalized fitnesses
    norm_fit = [fit / sum(fitnesses) for fit in fitnesses]
    cumulative_sum = np.array(norm_fit).cumsum()

    temp_print = print_crossover
    # Select the parents from the population
    # and then perform crossover
    mates = []
    children = []
    for idx in range(len(pop) // 2):
        mates.append([pop[roulette_wheel_selection(cumulative_sum, random.random())],
                      pop[roulette_wheel_selection(cumulative_sum, random.random())]])

        # If the two mates are the same individual,
        # try to replace one of the mates
        while np.array_equal(mates[idx][0], mates[idx][1]):
            mates[idx][1] = pop[roulette_wheel_selection(cumulative_sum, random.random())]

        if temp_print is True:
            print('\nbeginning crossover...')
            print('1st example to crossover:', mates[idx][0][0], mates[idx][0][1], mates[idx][0][2], mates[idx][0][3], mates[idx][0][4])
            print('1st example to crossover:', mates[idx][1][0], mates[idx][1][1], mates[idx][1][2], mates[idx][1][3], mates[idx][1][4])
        # Perform single-point crossover
        pivot = randint(2)
        children.append(np.append(mates[idx][0][0:pivot], mates[idx][1][pivot:]))
        children.append(np.append(mates[idx][1][0:pivot], mates[idx][0][pivot:]))
        if temp_print is True:
            print('1st example after crossover:', children[0][0], children[0][1], children[0][2], children[0][3], children[0][4])
            print('2nd example after crossover:', children[1][0], children[1][1], children[1][2], children[1][3], children[1][4])
        temp_print = False

    return children

def mutate(pop, mutation_rate, std_div, print_mutate):
    if print_mutate is True:
        print('\nbeginning mutation...')
        print('(some weights of) first member before mutation:', pop[0][0], pop[0][1], pop[0][2], pop[0][3], pop[0][4])
    mutated_pop = []
    for chromosome in pop:
        mutated_chrom = chromosome[:]
        for val_idx in range(len(chromosome)):
            rand_float = random.random()
            if rand_float <= mutation_rate:
                mutated_chrom[val_idx] += random.gauss(0, std_div)
        mutated_pop.append(mutated_chrom)

    if print_mutate is True:
        print('(some weights of) first member after mutation:', mutated_pop[0][0], mutated_pop[0][1], mutated_pop[0][2], mutated_pop[0][3], mutated_pop[0][4])
    return mutated_pop

def select(pop, ffnn, fit_fcn, num_chromosomes, print_selection):
    new_pop = []
    temp_print = print_selection
    for _ in range(num_chromosomes):
        new_pop.append(tournament_selection(pop, ffnn, fit_fcn, temp_print))
        temp_print = False
    return new_pop