""" -------------------------------------------------------------
@file        process_data.py
@authors     George Engel, Troy Oster, Dana Parker, Henry Soule
@brief       Imports and pre-processes data repositories
TODO: Make this a Class, update UML
"""

from database import database as db
import sys
import random
import os
import numpy as np
import shared_functions as sf

""" -------------------------------------------------------------
@param  input_database  The database file (of type .data) to be processed
@return     The pre-processed data from input_db as a database object
            (see database.py)
@brief      Loads the file contents into a database object
"""
def process_database_file(path_manager):

    # loads in data file from the selected database directory
    data_filename = path_manager.find_files(path_manager.get_current_selected_dir(), ".data")[0]
    full_data_path = os.path.join(path_manager.get_current_selected_dir(), data_filename)

    current_db_file = open(full_data_path, 'r')
    db_data = []

    for line in current_db_file:
        # Converts the string line from the file into a set of values.
        db_data.append(csv_to_set(line))
            
    current_db_file.close()
    
    if len(db_data[-1]) < 0:
        db_data.pop()
    elif db_data[-1][0] is "" and len(db_data[-1]) is 1: 
        db_data.pop()
        
    attributes, classifier_column, classifier_attr_cols, \
    missing_symbol, dataset_type, class_list, layers = read_attributes(path_manager.get_current_selected_dir(), data_filename)

    return db(db_data, attributes, classifier_column, classifier_attr_cols, missing_symbol, dataset_type, layers, class_list)

# Reads in the attribute file from a database, and returns the attributes as a list
def read_attributes(directory, data_filename):
    attribute_file_name = data_filename[:-4] + "attr"
    
    full_path = os.path.join(directory, attribute_file_name)
    
    attribute_file = open(full_path, 'r')
    
    # Reads in attributes from line 1 and split/cleans into list
    attributes = attribute_file.readline().strip('\n').split(',')
    
    # Reads in the index of the Classifier column.
    classifier_column = int(attribute_file.readline().strip('\n'))
    
    # Reads in the indexes of the attributes used for classification
    classifier_attr_cols = []
    for cols in  attribute_file.readline().strip('\n').split(','):
        classifier_attr_cols.append(int(cols))
        
    missing_symbol = attribute_file.readline().strip('\n')
    
    dataset_type = attribute_file.readline().strip('\n')
    
    class_list = []
    if dataset_type == 'classification':
        for c in attribute_file.readline().strip('\n').split(','):
            class_list.append(c)

    # Gets the layers of the nerual network from the attr configuration, with respective node counter per layer.
    layers = []
    for cols in  attribute_file.readline().strip('\n').split(','):
        layers.append(int(cols))
    
    return attributes, classifier_column, classifier_attr_cols, missing_symbol, dataset_type, class_list, layers

    
""" -------------------------------------------------------------
@param  input_csv   Comma-seperated string to convert
@return     A set of values found in input_csv
@brief      Converts comma seperated strings into a set of values
"""
def csv_to_set(input_csv):
    return input_csv.strip('\n').split(',')

# Goes over the database and runs necessary conversions.
def convert(database):
    if len(database) > 0:
        attribute_count = len(database[0])
        for attribute_col in range(0, attribute_count):
            if discrete_col_check(database, attribute_col):
                # Converts the attribute_col to a set of float values so they aren't strings. Thereby marking them as discrete.
                database = col_to_float(database, attribute_col)
    return database

# Returns true if the column is discreate
def discrete_col_check(database, attribute_col):
    for data_row in database:
        try:
            # Checks that the value for this row at position attribute_col is discrete
            float(data_row[attribute_col])
        except ValueError:
            # There was a non-discrete data type entered.
            return False
    return True

# Converts a column of strings into float format
def col_to_float(database, attribute_col):
    for data_row in database:
        data_row[attribute_col] = float(data_row[attribute_col])
    return database
        
""" -------------------------------------------------------------
@param  database           Input database to operate upon per @brief
@param  attribute_count    The amount of expected attributes for each row of data.
@return     input_db, correction_que :
            returns the 'clean' rows of data (input_db),
            and the rows of any malformed data (correction_queue).
@brief      Either removes data with missing parameters,
            or extrapolates missing data using bootstraping methodology.
"""
def data_correction(input_db, attribute_count):
    
    # Holds the rows of data that appear to be missing some attributes
    correction_queue = []
    
    # Checks if the data from a specific row
    # has all of the required parameters.
    # If not, pops it from the list into a later processing queue.
    for data in input_db.get_data():
        if len(data) is not attribute_count:
            correction_queue.append(data)
            input_db.remove(data)
            
    return input_db, correction_queue

# One hot encoding function for giving numerical value representation for catagorical values.
# def one_hot_encode(data_column):
#     # make a set of the data column list (Then convert to list again to index, because lazy)
#     # Should honestly use a hash table, but eh, thats effort.
#     catagorical_val_set = list(set(data_column))
    
#     # Build out one hot lists for each catcagorical value.
#     one_hot_set = []
#     for val in catagorical_val_set:
#         temp_list = [0] * len(catagorical_val_set)
#         temp_list[catagorical_val_set.index(val)] = 1
#         one_hot_set.append(temp_list)
    
#     one_hot_column = []
#     # Replace catagorical values with numerical indexes from the set
#     for data in data_column:
#         one_hot_column.append(one_hot_set[catagorical_val_set.index(data)])
        
#     return one_hot_column

# # Integer encoder
# def integer_encode(data_column):
#     # make a set of the data column list (Then convert to list again to index, because lazy)
#     # Should honestly use a hash table, but eh, thats effort.
#     #integer_set = list(set(data_column))
    
#     integer_column = []
#     # Replace catagorical values with numerical indexes from the set
#     for data in data_column:
#         integer_column.append(integer_set.index(data))
        
#     return integer_column

# Do the necessary encoding for the FFNN
def FFNN_encoding(db):

    # Find which attributes are numbers
    is_num = [True for attr in db.get_data()[0]]
    for idx, attr in enumerate(db.get_data()[0]):
        if is_num[idx] is True:
            try:
                temp = float(attr)
            except ValueError:
                is_num[idx] = False
    
    # Find all possible values of every categorical attribute
    possible_vals = [[] for attr in db.get_data()[0]]
    for ex in db.get_data():
        for idx, attr in enumerate(ex):
            if is_num[idx] is False or idx == db.get_classifier_col():
                if attr not in possible_vals[idx]:
                    possible_vals[idx].append(attr)
    
    # Sort the possible values so it's easier to read
    possible_vals = [sorted(attr) if len(attr) != 0 else [] for attr in possible_vals]
    
    # One-hot encode the classifier attribute
    # Store the possible classifer values
    # Integer encode the non-classifier attributes
    new_data = []
    for ex in db.get_data():
        new_ex = []
        encode = []
        for attr_idx, attr in enumerate(ex):
            if len(possible_vals[attr_idx]) == 0:
                new_ex.append(np.array([attr], dtype=np.float32))
            else:

                # One-hot encoding
                if attr_idx == db.get_classifier_col():
                    if db.get_dataset_type() == 'classification':
                        temp = np.asarray([1 if attr == val else 0 for val in possible_vals[attr_idx]])
                        encode = np.zeros((len(possible_vals[attr_idx]), 1))
                        encode[np.argmax(temp)] = 1
                    elif db.get_dataset_type() == 'regression':
                        encode = np.asarray(sf.sigmoid(attr))
                
                # Integer encoding
                else:
                    new_ex.append(np.array([possible_vals[attr_idx].index(attr)], dtype=np.float32))

        new_data.append((sf.sigmoid(np.asarray(new_ex)), encode))
    
    db.set_data(new_data)
    db.set_class_list(possible_vals[db.get_classifier_col()])

# Finds any ambiguous/missing data and returns the rows of the relevant database in which missing parameters occur.
def identify_missing_data(input_db):
    
    # Holds the rows of data that appear to be missing some attributes
    correction_queue = []
    normal = []
    for data in input_db.get_data():
        
        # Check for a missing parameter character:
        if input_db.get_missing_symbol() in data:
            # Adds data to que for correction
            correction_queue.append(data)
        else:
            # Adds data to normal database (won't have to be modified further via extrapolation)
            normal.append(data)
            
    return normal, correction_queue

def extrapolate_data(normal_data, malformed_data, missing_data_val):
    corrected_data = []
    
    for data in malformed_data:
        corrected_data.append(bootstrap_selection(normal_data, data, missing_data_val))
        
    return corrected_data
    
# Fills in the unknown/missing data from existing normal data randomly.
def bootstrap_selection(normal_data, malformed_row, missing_data_val):
    length = len(malformed_row)
    
    corrected_data = []
    
    for index in range(length):
        if malformed_row[index] is "?":
            corrected_data.append(random.choice(normal_data)[index])
        else:
            corrected_data.append(malformed_row[index])
            
    return corrected_data


def shuffle_all(training_data, percent):
    return_data = training_data
    if len(return_data) > 0:
        attribute_count = 0
        for attribute in return_data[0]:
            shuffle_data(return_data, percent, attribute_count)
            attribute_count += 1
    return return_data

# Returns a percentage of the dataset that is determined at random
def random_data_from(dataset, percent):
    random_dataset = []

    if(percent <= 100):
        num_to_shuffle = int(len(dataset) * percent)
        
        # Will need to pull X% of the rows out of the database at random
        for iter in range(0, num_to_shuffle):
            data_row = random.choice(dataset)
            dataset.remove(data_row)
            
            # Adds the row data to random dataset.
            random_dataset.append(data_row)
    return random_dataset
  
# Shuffles X% of the data for an attribute specified by row of dataset.
def shuffle_data(training_data, percent, attribute):
    shuffling = []

    if(percent <= 100):
        num_to_shuffle = int(len(training_data) * percent)
        
        # Will need to pull X% of the rows out of the database at random
        for iter in range(0, num_to_shuffle):
            data_row = random.choice(training_data)
            training_data.remove(data_row)
            
            # Adds the row data to be shuffled.
            shuffling.append(data_row)
    
# Shuffles X% of the data for an attribute specified by row of dataset.
def shuffle_data(training_data, percent, attribute):
    shuffling = []

    if(percent <= 100):
        num_to_shuffle = int(len(training_data) * percent)
        
        # Will need to pull X% of the rows out of the database at random
        for iter in range(0, num_to_shuffle):
            data_row = random.choice(training_data)
            training_data.remove(data_row)
            
            # Adds the row data to be shuffled.
            shuffling.append(data_row)
        
        # Now we shift all the attribute's in the column specified down one from the randomly generated shuffle list.
        last_attribute = shuffling[-1][attribute]
        temp_attribute = shuffling[0][attribute]
        
        # Sets the first row's attribute equal to the last row's.
        shuffling[0][attribute] = last_attribute
        training_data.append(shuffling.pop(0))
        
        # Shuffles remaining data
        for data_row in shuffling:
            temp = data_row[attribute]
            data_row[attribute] = temp_attribute
            temp_attribute = temp
        
        training_data += shuffling
        # print("Data Shuffled by",(percent*100),"%")
    else:
        print("ERROR: can't shuffle more than the size of the database.")

# Isn't this just width based binning?
# Prepare data for k-fold
# Moved this function over from classifier.py in assignment 1
def separate_data(attributes, data):
     binSize = int(len(data)/10)
     bin_lengths = []
     row_idx = 0
     return_data = []
     for index in range(10):
         bin_lengths.append(binSize)
     for index in range((len(data)%10 )):
         bin_lengths[index] += 1
     for bin_idx in range(len(bin_lengths)):
        for row in range(bin_lengths[bin_idx]):
            example = data[row_idx]
            return_data.append([bin_idx, example])
            row_idx += 1
     return [return_data,bin_lengths]
