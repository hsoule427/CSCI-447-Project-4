""" -------------------------------------------------------------
@brief  This stores some of the functionality for selecting a database
        and preparing a db object that we had in main.py in previous projects
"""
import process_data


def select_db(databases):  
    if len(databases) == 0:
        print("ERROR: No databases found!")
        return False
    chosen = False
    db = ""
    chosen_dbs = []
    # Selection loop for database
    while(not chosen):
        print("\nEnter one of the databases displayed, or 'all' to run for all avalible databases.:", databases)
        db = input("Entry: ")
        print("database:", db)
        if db in databases:
            print("Selected:", db)
            chosen_dbs.append(db)
            chosen = True
        elif db.lower() == "all":
            print("Running for all Databases.")
            chosen_dbs = ["abalone", "car", "forestfires", "machine", "segmentation", "wine"]
            chosen = True
        else:
            print(db, "is an invalid entry. Try again.")
    return chosen_dbs


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
    
    # output_file.write('CURRENT DATASET: ' + database + '\n')
    # debug_file.write('CURRENT DATASET: ' + database + '\n')
    # output_file.write('DATA TYPE: ' + db.get_dataset_type() + '\n')
    # debug_file.write('DATA TYPE: ' + db.get_dataset_type() + '\n')
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
    # TODO: make it append the database name to the debug file aswell, so we can get every dataset when running for all of them.
    # debug_file.write('\n\nFULL DATASET: \n')
    # for row in db.get_data():
    #     debug_file.write(str(row) + '\n')
    
    return db

# -------------------------------------------------------------
# Cleaner print outs for the sake of my sanity.
def print_db(db):
    if len(db) < 1:
        print("[] - Empty")
    else:
        for row in db:
            print(row)