"""  
@file       database.py
@authors    George Engel, Troy Oster, Dana Parker, Henry Soule
@brief      Object that stores the information of
            each data repository for ease of access & manipulation
"""

import random
import process_data

class database:
    """
    @param  data_array  List of data from one data repository
                        that will be or has been filtered.
    """
    def __init__(self, data_array, attrs, classifier_col, classifier_attr_cols, symbol, dataset_type, layers, class_list=[]):
        print("Database initialized.")
        self.data = data_array
        self.attributes = attrs
        self.classifier_column = classifier_col
        self.classifier_attr_columns = classifier_attr_cols
        self.missing_symbol = symbol
        self.db_type = dataset_type
        self.class_list = class_list
        self.layers = layers

        
    def convert_discrete_to_float(self):
        self.data = process_data.convert(self.data)
        # print("SELF DATA:", self.data)
        
    def get_layers(self):
        return self.layers  
    
    def to_string(self):
        if len(self.data) < 1:
            print("[] - Empty")
        else:
            for row in self.data:
                print(row)
        
    def get_data(self):
        return self.data
    
    def set_data(self, data_array):
        self.data = data_array
        
    def get_attr(self):
        return self.attributes
    
    def get_classifier_col(self):
        return self.classifier_column
    
    def get_classifier_attr_cols(self):
        return self.classifier_attr_columns
    
    def set_classifier_attr_cols(self, attr_cols):
        self.classifier_attr_columns = attr_cols
    
    def get_missing_symbol(self):
        return self.missing_symbol
    
    def get_dataset_type(self):
        return self.db_type
    
    def get_classifiers(self):
        class_idx = self.get_classifier_col()
        classifiers = []
        for row in self.data:
            if row[class_idx] not in classifiers:
                classifiers.append(row[class_idx])
        
        return classifiers
    
    def get_class_list(self):
        return self.class_list

    def set_class_list(self, vals):
        self.class_list = vals
    
    def get_training_data(self, start_idx, end_idx):
        return self.data[start_idx : end_idx]
