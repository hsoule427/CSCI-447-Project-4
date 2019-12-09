""" -------------------------------------------------------------
@file        pso_driver.py
@brief       A file for testing our pso implementation
"""

import process_data
from path_manager import pathManager as path_manager
import numpy as np
import prepare_data
import os.path
import shared_functions as sf
import gen_algo as ga
from FFNN import FFNN
import sys


pm = path_manager()
selected_dbs = prepare_data.select_db(pm.find_folders(pm.get_databases_dir()))
db = prepare_data.prepare_db(selected_dbs[0], pm)

# BEGIN classification FFNN
if db.get_dataset_type() == 'classification':

    # BEGIN preprocessing
    process_data.FFNN_encoding(db)

    # ----------------------------------------------------------------------------------
    # Run the GA

    layer_sizes = [len(db.get_attr()) - 1, 50, 50, len(db.get_class_list())]  # (3)
    learning_rate = 2
    num_chromosomes = 20
    num_generations = 10
    mutation_rate = .5
    mutation_std_div = .5

    # Initialize the population
    curr_pop = ga.init_population(sf.calc_total_vec_length(layer_sizes), num_chromosomes)

    # Evalutate the initial fitness
    ffnn = FFNN.init_no_weights(db.get_data(), learning_rate, layer_sizes)
    curr_fitness = []
    for curr_chrom in curr_pop:
        w_vec, b_vec = sf.encode_weight_and_bias(curr_chrom, ffnn.layer_sizes)
        ffnn.set_weight(w_vec)
        ffnn.set_biases(b_vec)
        num_correct, total = ffnn.zero_one_loss()
        curr_fitness.append(num_correct / total)

    # Create list of tuples of population and fitness and sort by fitness
    zipped_curr_pop = sorted(zip(curr_pop, curr_fitness), key=lambda x: x[1])
    print('Best fitness in initial population ——>', zipped_curr_pop[0][1])

    for t in range(num_generations):

        # Select members the new population
        new_pop = ga.select(curr_pop, ffnn, sf.classification_error, num_chromosomes)

        # Update the fitnesses of the new population
        new_fitness = []
        for curr_chrom in new_pop:
            w_vec, b_vec = sf.encode_weight_and_bias(curr_chrom, ffnn.layer_sizes)
            ffnn.set_weight(w_vec)
            ffnn.set_biases(b_vec)
            num_correct, total = ffnn.zero_one_loss()
            new_fitness.append(num_correct / total)

        # Pair up members to crossover
        new_pop = ga.crossover(new_pop, new_fitness)
        new_pop = ga.mutate(new_pop, mutation_rate, mutation_std_div)

        # Update the fitnesses of the new population
        new_fitness = []
        for curr_chrom in new_pop:
            w_vec, b_vec = sf.encode_weight_and_bias(curr_chrom, ffnn.layer_sizes)
            ffnn.set_weight(w_vec)
            ffnn.set_biases(b_vec)
            num_correct, total = ffnn.zero_one_loss()
            new_fitness.append(num_correct / total)

        # Create list of tuples of population and fitness and sort by fitness
        zipped_new_pop = sorted(zip(new_pop, new_fitness), key=lambda x: x[1])
        print('Best fitness in generation', t + 1, '——>', zipped_new_pop[0][1])
        curr_pop = new_pop
        # curr_pop = [
        #     [new_chromosome[0] for new_chromosome in zipped_new_pop[0:len(zipped_new_pop) // 2]].append(
        #     [curr_chromosome[0] for curr_chromosome in zipped_curr_pop[0:len(zipped_curr_pop) // 2]])]
        curr_fitness = new_fitness
        # zipped_curr_pop = zipped_new_pop

# BEGIN regression FFNN
elif db.get_dataset_type() == 'regression':

    # BEGIN preprocessing
    process_data.FFNN_encoding(db)

    # ----------------------------------------------------------------------------------
    # Run the GA

    layer_sizes = [len(db.get_attr()) - 1, 50, 50, 1]  # (3)
    learning_rate = 5
    num_chromosomes = 20
    num_generations = 10
    mutation_rate = .5
    mutation_std_div = .5

    # Initialize the population
    curr_pop = ga.init_population(sf.calc_total_vec_length(layer_sizes), num_chromosomes)

    # Evalutate the initial fitness
    ffnn = FFNN.init_no_weights(db.get_data(), learning_rate, layer_sizes)
    curr_fitness = []
    for curr_chrom in curr_pop:
        w_vec, b_vec = sf.encode_weight_and_bias(curr_chrom, ffnn.layer_sizes)
        ffnn.set_weight(w_vec)
        ffnn.set_biases(b_vec)
        error = ffnn.regression_error()
        curr_fitness.append(error)

    # Create list of tuples of population and fitness and sort by fitness
    zipped_curr_pop = sorted(zip(curr_pop, curr_fitness), key=lambda x: x[1])[::-1]
    print('Best fitness in initial population ——>', zipped_curr_pop[0][1])

    for t in range(num_generations):

        # Select members the new population
        new_pop = ga.select(curr_pop, ffnn, sf.classification_error, num_chromosomes)

        # Update the fitnesses of the new population
        new_fitness = []
        for curr_chrom in new_pop:
            w_vec, b_vec = sf.encode_weight_and_bias(curr_chrom, ffnn.layer_sizes)
            ffnn.set_weight(w_vec)
            ffnn.set_biases(b_vec)
            error = ffnn.regression_error()
            new_fitness.append(error)

        # Pair up members to crossover
        new_pop = ga.crossover(new_pop, new_fitness)
        new_pop = ga.mutate(new_pop, mutation_rate, mutation_std_div)

        # Update the fitnesses of the new population
        new_fitness = []
        for curr_chrom in new_pop:
            w_vec, b_vec = sf.encode_weight_and_bias(curr_chrom, ffnn.layer_sizes)
            ffnn.set_weight(w_vec)
            ffnn.set_biases(b_vec)
            error = ffnn.regression_error()
            new_fitness.append(error)

        # Create list of tuples of population and fitness and sort by fitness
        zipped_new_pop = sorted(zip(new_pop, new_fitness), key=lambda x: x[1])[::-1]
        print('Best fitness in generation', t + 1, '——>', zipped_new_pop[0][1])
        curr_pop = new_pop
        # curr_pop = [
        #     [new_chromosome[0] for new_chromosome in zipped_new_pop[0:len(zipped_new_pop) // 2]].append(
        #     [curr_chromosome[0] for curr_chromosome in zipped_curr_pop[0:len(zipped_curr_pop) // 2]])]
        curr_fitness = new_fitness
        # zipped_curr_pop = zipped_new_pop

else:
    print('Database type invalid. Type = ' + db.get_dataset_type())