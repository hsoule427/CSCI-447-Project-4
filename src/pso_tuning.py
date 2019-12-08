import process_data
import Cost_Functions as cf
from FFNN import FFNN
from path_manager import pathManager as path_manager
import numpy as np
import os.path
import prepare_data
import shared_functions as sf
import pso

# Create all permutations for hyperparams
# each param is a different set
def make_grid(param1, param2, param3):
    params = []
    for a in param1:
        for b in param2:
            for c in param3:
                params.append([a,b,c])
    return params



pm = path_manager()
selected_dbs = prepare_data.select_db(pm.find_folders(pm.get_databases_dir()))
db = prepare_data.prepare_db(selected_dbs[0], pm)
process_data.shuffle_all(db.get_data(), 1)

tuning_file = open('./tuning_files/pso_tuning_' + selected_dbs[0] + '.txt', 'w')

# hyperparamters
# (1) c1 - first acceleration coefficient
# (2) c2 - second acceleration coefficient
# (3) inertia
hp = [
    [0.5, 1.0, 1.5],                #(1)
    [0.5, 1.0, 1.5],                #(2)
    [0.01, 0.1, 0.3, 0.5, 0.7, 0.9] #(3)
]

permutations = make_grid(hp[0], hp[1], hp[2])

if db.get_dataset_type() == 'classification':

    # BEGIN preprocessing
    process_data.FFNN_encoding(db)
    

    # (1) First layer (input layer) has 1 node per attribute.
    # (2) Hidden layers has arbitrary number of nodes.
    # (3) Output layer has 1 node per possible classification.
    layer_sizes = [len(db.get_attr())-1,          # (1)
                    5, 5,                       # (2)
                    len(db.get_class_list())]   # (3)
    
    learning_rate = 1.5
    end_idx = int(len(db.get_data())*.33)
    
    tuning_file.write('PSO TUNING\n')
    tuning_file.write('CURRENT DATABASE: ' + selected_dbs[0] + '\n')
    # Loop thru each permutation of our hyperparameters
    for i,perm in enumerate(permutations):
        print("PERMUTATION ", i+1, "/", len(permutations))
        tuning_file.write('CURRENT PERMUTATION: ' + str(perm) + '\n')
        fitness, avg_distance = pso.main_loop(db.get_data()[0:end_idx], layer_sizes, learning_rate, hp)
        tuning_file.write('FINAL FITNESS: ' + str(fitness) + '\n')
        tuning_file.write('FINAL AVG DISTANCE: ' + str(avg_distance) + '\n')
        tuning_file.write('-----------------------------------\n')



# BEGIN regression FFNN
elif db.get_dataset_type() == 'regression':

    # (1) First layer (input layer) has 1 node per attribute.
    # (2) Hidden layers has arbitrary number of nodes.
    # (3) Output layer has 1 node, just some real number.
    layer_sizes = [
        len(db.get_attr()), # (1)
        5, 5,               # (2)
        1                   # (3)
    ]

    

else:
    print('Database type invalid. Type = ' + db.get_dataset_type())
    

    





