import process_data
import Cost_Functions as cf
from FFNN import FFNN
from path_manager import pathManager as path_manager
import numpy as np
import os.path
import prepare_data
import shared_functions as sf
import diff_evo

pm = path_manager()
selected_dbs = prepare_data.select_db(pm.find_folders(pm.get_databases_dir()))


db = prepare_data.prepare_db(selected_dbs[0], pm)

if db.get_dataset_type() == 'classification':

    # BEGIN preprocessing
    process_data.FFNN_encoding(db)
    

    # (1) First layer (input layer) has 1 node per attribute.
    # (2) Hidden layers has arbitrary number of nodes.
    # (3) Output layer has 1 node per possible classification.
    
    layer_sizes = [len(db.get_attr()),          # (1)
                    5, 5,                       # (2)
                    len(db.get_class_list())]   # (3)

    # This number is arbitrary.
    # TODO: Tune this per dataset
    learning_rate = 1.5

    diff_evo.main_loop(db, layer_sizes, learning_rate, generations=1000)



    

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