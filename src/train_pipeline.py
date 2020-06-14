import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score


class Pipeline:

    def __init__(self, features, target, numerical_to_float, numerical_to_impute, numerical_to_scale,
                 categorical_with_missing, categorical_with_rare, dict_of_freq_labels,
                 test_size=0.2, random_state=0):

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.features = features
        self.target = target

        self.numerical_to_float = numerical_to_float
        self.numerical_to_impute = numerical_to_impute
        self.numerical_to_scale = numerical_to_scale
        self.categorical_with_missing = categorical_with_missing
        self.categorical_with_rare = categorical_with_rare
        self.dict_of_freq_labels = dict_of_freq_labels
        self.dict_of_impute_values = {}

        self.test_size = test_size
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = LogisticRegression(self.random_state)

    '''Preprocessing functions:'''


    '''Master Functions'''

    def fit(self):
        '''Will fit the model INCLUDING a train/test split on the data.'''
        return

    def __transform(self):
        '''Will transform a dataset to be ready for prediction.'''
        return

    def predict(self):
        '''Will predict on new data.'''
        return

    def evaluate_model(self):
        return



