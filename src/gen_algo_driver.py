""" -------------------------------------------------------------
@file        pso_driver.py
@brief       A file for testing our pso implementation
"""

import process_data
from path_manager import pathManager as path_manager
import numpy as np
import prepare_data
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

    # (1) First layer (input layer) has 1 node per attribute.
    # (2) Hidden layers has arbitrary number of nodes.
    # (3) Output layer has 1 node per possible classification.

    layer_sizes = [len(db.get_attr()) - 1,  # (1)
                   5, 5,  # (2)
                   len(db.get_class_list())]  # (3)

    # This number is arbitrary.
    # TODO: Tune this per dataset
    learning_rate = 1.5

    # ----------------------------------------------------------------------------------
    # Run the GA

    num_chromosomes = 20
    num_generations = 20
    mutation_rate = 0.015
    mutation_std_div = 0.001

    # Initialize the population
    curr_pop = ga.init_population(sf.calc_total_vec_length(layer_sizes), num_chromosomes)

    # Evalutate the initial fitness
    ffnn = FFNN.init_no_weights(db.get_data(), learning_rate, layer_sizes)
    curr_fitness = []
    for curr_chrom in curr_pop:
        w_vec, b_vec = sf.encode_weight_and_bias(curr_chrom, ffnn.layer_sizes)
        ffnn.set_weight(w_vec)
        ffnn.set_biases(b_vec)
        curr_fitness.append(ffnn.get_fitness(sf.classification_error))

    for _ in range(num_generations):

        # Select new members of the population
        new_pop = ga.select(curr_pop, ffnn, sf.classification_error, num_chromosomes)

        # Update the fitnesses of the current population
        curr_fitness = []
        for curr_chrom in new_pop:
            w_vec, b_vec = sf.encode_weight_and_bias(curr_chrom, ffnn.layer_sizes)
            ffnn.set_weight(w_vec)
            ffnn.set_biases(b_vec)
            curr_fitness.append(ffnn.get_fitness(sf.classification_error))

        # Pair up members to crossover
        new_pop = ga.crossover(curr_pop, curr_fitness)
        new_pop = ga.mutate(curr_pop, mutation_rate, mutation_std_div)

    # TODO delete when necessary
    # print('\n\n\nTEST')
    # print(sf.encode_weight_and_bias(best_chromosome, layer_sizes))
    # print('\nfitness')
    # print(ffnn.get_fitness(sf.classification_error))
    # sys.exit()



# BEGIN regression FFNN
elif db.get_dataset_type() == 'regression':

    # (1) First layer (input layer) has 1 node per attribute.
    # (2) Hidden layers has arbitrary number of nodes.
    # (3) Output layer has 1 node, just some real number.
    layer_sizes = [
        len(db.get_attr()) - 1,  # (1)
        5, 5,  # (2)
        1  # (3)
    ]

else:
    print('Database type invalid. Type = ' + db.get_dataset_type())