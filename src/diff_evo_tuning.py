import process_data
import Cost_Functions as cf
from FFNN import FFNN
from path_manager import pathManager as path_manager
import numpy as np
import os.path
import prepare_data
import shared_functions as sf
import diff_evo

def make_grid(param1, param2):
    params = []
    for a in param1:
        for b in param2:
            params.append([a,b])
    return params

pm = path_manager()
selected_dbs = prepare_data.select_db(pm.find_folders(pm.get_databases_dir()))
db = prepare_data.prepare_db(selected_dbs[0], pm)
process_data.shuffle_all(db.get_data(), 1)

tuning_file = open('./tuning_files/diff_evo_tuning_' + selected_dbs[0] + '.txt', 'w')

# Set of tunable hyperparameters
# (1) Beta value
# (2) Probability
hp = [
    [0.5, 1.0, 1.5, 1.8],    # (1)
    [0.3, 0.5, 0.7]     # (2)
]

permutations = make_grid(hp[0], hp[1])


if db.get_dataset_type() == 'classification':

    # BEGIN preprocessing
    process_data.FFNN_encoding(db)
    

    # (1) First layer (input layer) has 1 node per attribute.
    # (2) Hidden layers has arbitrary number of nodes.
    # (3) Output layer has 1 node per possible classification.
    
    layer_sizes = [len(db.get_attr())-1,          # (1)
                    5, 5,                       # (2)
                    len(db.get_class_list())]   # (3)

    # This number is arbitrary.
    # TODO: Tune this per dataset
    learning_rate = 1.5
    end_idx = int(len(db.get_data())*.33)

    for i,perm in enumerate(permutations):
        print("PERMUTATION ", i+1, "/", len(permutations), ": ", perm)
        tuning_file.write(str(perm) + '\n')
        fitness, avg_dist = diff_evo.main_loop(db.get_data()[0:end_idx], db.get_dataset_type(), \
                                                layer_sizes, learning_rate, perm)
        tuning_file.write('FINAL FITNESS: ' + str(fitness) + '\n')
        tuning_file.write('FINAL AVG DISTANCE: ' + str(avg_dist) + '\n')
        tuning_file.write('-----------------------------------\n')




    

# BEGIN regression FFNN
elif db.get_dataset_type() == 'regression':

    process_data.FFNN_encoding(db)

    print(db.get_data())
    
    learning_rate = 1.5

    # (1) First layer (input layer) has 1 node per attribute.
    # (2) Hidden layers has arbitrary number of nodes.
    # (3) Output layer has 1 node, just some real number.
    layer_sizes = [
        len(db.get_attr())-1, # (1)
        5, 5,               # (2)
        1                   # (3)
    ]

    # ffnn = FFNN(db.get_data()[0], learning_rate)
    
    # Make weights and biases
    # w_set = np.random.randn(sf.calc_total_vec_length(layer_sizes))
    # weights, biases = sf.encode_weight_and_bias(w_set, layer_sizes)
    # ffnn.set_weight(weights)
    # ffnn.set_biases(biases)

    # ffnn.get_fitness()

    


else:
    print('Database type invalid. Type = ' + db.get_dataset_type())