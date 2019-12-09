import process_data
import Cost_Functions as cf
from FFNN import FFNN
from path_manager import pathManager as path_manager
import numpy as np
import sys
import os.path
import prepare_data
import shared_functions as sf
import diff_evo

pm = path_manager()
selected_dbs = prepare_data.select_db(pm.find_folders(pm.get_databases_dir()))
db = prepare_data.prepare_db(selected_dbs[0], pm)
process_data.shuffle_all(db.get_data(), 1)
half_idx = int(len(db.get_data()) * .5) # Get index of midpoint of dataset

if db.get_dataset_type() == 'classification':
    # Encode data
    process_data.FFNN_encoding(db)
    # (1) First layer (input layer) has 1 node per attribute.
    # (2) Hidden layers has arbitrary number of nodes.
    # (3) Output layer has 1 node per possible classification.
    

    layer_sizes = [len(db.get_attr())-1, 5, 5, len(db.get_class_list())]

    # This number is arbitrary.
    # TODO: Tune this per dataset
    learning_rate = 1.5


# BEGIN regression FFNN
elif db.get_dataset_type() == 'regression':
    # Encode data
    process_data.FFNN_encoding(db)    
    learning_rate = 1.5
    # (1) First layer (input layer) has 1 node per attribute.
    # (2) Hidden layers has arbitrary number of nodes.
    # (3) Output layer has 1 node, just some real number.
    
    layer_sizes = [len(db.get_attr())-1, 5, 5, 1]
        
    

else:
    print('Database type invalid. Type = ' + db.get_dataset_type())
    sys.exit()


# TRAIN
start_fts, end_fts, best_weights = diff_evo.main_loop(db.get_data(), db.get_dataset_type(), \
                                                        layer_sizes, learning_rate, [0.1, 0.7], generations=1)


# rate = sf.compute_rate(start_fts, end_fts, 100)


#VALIDATE
# ffnn = FFNN.init_no_weights(db.get_data()[half_idx:len(db.get_data())], learning_rate)
# weights, biases = sf.encode_weight_and_bias(best_weights, layer_sizes)
# ffnn.set_biases(biases)
# ffnn.set_weight(weights)
# if db.get_dataset_type() == 'classification':
#     fitness = ffnn.zero_one_loss()
# else:
#     fitness = ffnn.regression_error()


