""" -------------------------------------------------------------
@file        main.py
@authors     George Engel, Troy Oster, Dana Parker, Henry Soule
@brief       The file that runs the program
"""

import sys

# -------------------------------------------------------------
# Third-party imports

import numpy as np
import os.path

# -------------------------------------------------------------
# Custom imports

import process_data
from FFNN import FFNN
from path_manager import pathManager as path_manager
import get_dbs as GDB

# -------------------------------------------------------------

def select_db(databases):
    if len(databases) == 0:
        print("ERROR: No databases found!")
        return False
    chosen = False
    db_name = ""
    chosen_db = databases[0]
    # Selection loop for database
    while(not chosen):
        print("\nEnter one of the databases displayed:", databases)
        db_name = input("Entry: ")
        print("database:", db_name)
        if db_name in databases:
            print("Selected:", db_name)
            chosen_db = db_name
            chosen = True
        else:
            print(db_name, "is an invalid entry. Try again.")
    return chosen_db


# Set the path manager's current save folder to the current settings if it exists, or create it
def verify_save_folder(pm, db):
    save_folder_name = ""
    for layer in db.get_layers():
        save_folder_name+=str(layer)+"-"

    # Remove the last extra character
    save_folder_name = save_folder_name[:-1]

    pm.set_save_state_folder(save_folder_name)

    pm.make_folder_at_dir(pm.get_save_state_dir())

    return select_save_state(pm)


# Allows you to load from a save state for a given database's layer/node combination if one is present.
def select_save_state(pm):
    # Checks that save state folder contains states
    if len(pm.find_files(pm.get_save_state_dir(), "")) > 0:

        # Provides the option to load from an existing save state
        awns = input("\nWould you like to load from an existing save state?")
        if awns.lower() is "y":
            exists = True
            while(exists):
                print("Current save states (Epochs):", pm.find_files(pm.get_save_state_dir(), ""))
                save_state = input("Select a saved state (Epoch #):")
                # Validate the save state exists
                path = os.path.join(pm.get_save_state_dir(), save_state)
                exists = pm.validate_file(path)

                if exists:
                    # Load save_state object and return!
                    return ss.load_state(path)
                else:
                    print("Invalid save state. Try again.")
        else:
            print("Beginning new Neural Net...")
    return False

# -------------------------------------------------------------

def prepare_db(database, pm):
    # Sets the selected database folder
    # in the path manager for referencing via full path.
    pm.set_current_selected_folder(database)
    # Processes the file path of the database into
    # a pre processed database ready to be used as a learning/training set.
    db = process_data.process_database_file(pm)

    save_state = verify_save_folder(pm, db)

    if save_state is not False:
        # This is where we use the loaded save state object specified
        pass

    # Sanity checks.
    normal_data, irregular_data = process_data.identify_missing_data(db)
    corrected_data = process_data.extrapolate_data(normal_data, irregular_data, db.get_missing_symbol())
    # repaired_db is the total database once the missing values have been filled in.
    if len(corrected_data) > 0:
        repaired_db = normal_data + corrected_data
    else:
        repaired_db = normal_data

    db.set_data(repaired_db)
    # Convert the discrete data to type float.
    db.convert_discrete_to_float()

    return db

# -------------------------------------------------------------
# Cleaner print outs for the sake of my sanity.
def print_db(db):
    if len(db) < 1:
        print("[] - Empty")
    else:
        for row in db:
            print(row)

# -------------------------------------------------------------
def main():
    pm = path_manager()
    
    print("LEN:",len(sys.argv))
    # If a database has been passed in via commandline arguements, use that instead of asking user which to use.
    selected_db = select_db(GDB.get_db())

    db = prepare_db(selected_db, pm)

        # BEGIN classification FFNN
    if db.get_dataset_type() == 'classification':

        # BEGIN preprocessing
        process_data.FFNN_encoding(db)

        # (1) First layer (input layer) has 1 node per attribute.
        # (2) Hidden layers has arbitrary number of nodes.
        # (3) Output layer has 1 node per possible classification.
        layer_sizes = [len(db.get_attr()) - 1, 50, len(db.get_class_list())]

        learning_rate = float(1/10)
        ffnn = FFNN(layer_sizes, db.get_data(), db.get_dataset_type(), learning_rate)

    # BEGIN regression FFNN
    elif db.get_dataset_type() == 'regression':

        process_data.FFNN_encoding(db)

        # (1) First layer (input layer) has 1 node per attribute.
        # (2) Hidden layers has arbitrary number of nodes.
        # (3) Output layer has 1 node, just some real number.
        layer_sizes = [len(db.get_attr()) - 1, 50, 1]

        learning_rate = float(1/10)

        ffnn = FFNN(layer_sizes, db.get_data(), db.get_dataset_type(), learning_rate)

    else:
        print('Database type invalid. Type = ' + db.get_dataset_type())


# -------------------------------------------------------------
def main(db, learning_rate):
    pm = path_manager()

    db = prepare_db(db, pm)

    # BEGIN classification FFNN
    if db.get_dataset_type() == 'classification':

        # BEGIN preprocessing
        process_data.FFNN_encoding(db)

        # (1) First layer (input layer) has 1 node per attribute.
        # (2) Hidden layers has arbitrary number of nodes.
        # (3) Output layer has 1 node per possible classification.
        layer_sizes = [len(db.get_attr()) - 1, 50, len(db.get_class_list())]
        
        ffnn = FFNN(layer_sizes, db.get_data(), db.get_dataset_type(), learning_rate)
        return ffnn.zero_one_loss()
    # BEGIN regression FFNN
    elif db.get_dataset_type() == 'regression':

        process_data.FFNN_encoding(db)

        # (1) First layer (input layer) has 1 node per attribute.
        # (2) Hidden layers has arbitrary number of nodes.
        # (3) Output layer has 1 node, just some real number.
        layer_sizes = [len(db.get_attr()) - 1, 50, 1]

        ffnn = FFNN(layer_sizes, db.get_data(), db.get_dataset_type(), learning_rate)
        return ffnn.regression_error()
    else:
        print('Database type invalid. Type = ' + db.get_dataset_type())

# -------------------------------------------------------------

if __name__ == '__main__':
    main()