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

def tournament_selection(pop, ffnn, fit_fcn,
                       k=2):
    selections = []
    idx = []

    # Select the (non-duplicate) individuals randomly
    while len(selections) != k:
        rand_idx = random.randint(0, len(pop) - 1)
        if rand_idx not in idx:
            idx.append(rand_idx)
            selections.append(pop[rand_idx])

    # Evaluate the fitness of the selections
    fitnesses = []
    for curr_chrom in selections:
        w_vec, b_vec = sf.encode_weight_and_bias(curr_chrom, ffnn.layer_sizes)
        ffnn.set_weight(w_vec)
        ffnn.set_biases(b_vec)
        fitnesses.append(ffnn.get_fitness(fit_fcn))

    # Return the individual with the lower (better) fitness
    return selections[fitnesses.index(min(fitnesses))]

def roulette_wheel_selection(cumulative_sum, prob):
    rv = list(cumulative_sum)
    rv.append(prob)
    rv = sorted(rv)
    return rv.index(prob)


def crossover(pop, fitnesses):

    # Get the normalized fitnesses
    norm_fit = [fit / sum(fitnesses) for fit in fitnesses]
    cumulative_sum = np.array(norm_fit).cumsum()

    # Select the parents from the population
    # and then perform crossover
    mates = []
    children = []
    for idx in range(len(pop) // 2):
        mates.append([pop[roulette_wheel_selection(cumulative_sum, random.random())],
                      pop[roulette_wheel_selection(cumulative_sum, random.random())]])

        # If the two mates are the same individual,
        # try to replace one of the mates
        while mates[idx][0] == mates[idx][1]:
            mates[idx][1] = pop[roulette_wheel_selection(cumulative_sum, random.random())]

        # Perform single-point crossover
        pivot = randint(1, len(mates[idx][0]) - 2)
        children.append([mates[idx][0][0:pivot] + mates[idx][1][pivot:]])
        children.append(mates[idx][1][0:pivot] + mates[idx][0][pivot:])

    return children

def mutate(pop, mutation_rate, std_div):
    mutated_pop = []
    for chromosome in pop:
        mutated_chrom = chromosome[:]
        for val_idx in range(len(chromosome)):
            rand_float = random.random(0, 1)
            if rand_float <= mutation_rate:
                mutated_chrom[val_idx] += random.gauss(0, std_div)
        mutated_pop.append(mutated_chrom)
    return mutated_pop

def select(pop, ffnn, fit_fcn, num_chromosomes):
    new_pop = []
    for _ in range(num_chromosomes):
        new_pop.append(tournament_selection(pop, ffnn, fit_fcn))
    return new_pop