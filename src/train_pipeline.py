import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

import joblib   # only necessary if saving the model.


class TrainingPipeline:

    def __init__(self, features, target, numerical_to_impute, numerical_to_scale, categorical_with_missing,
                 categorical_with_rare, dict_of_freq_labels, test_size=0.2, random_state=0, save_train_flag=True,
                 save_train_location='./train.csv', save_test_flag=True, save_test_location='./test.csv',
                 save_model_flag=True, save_model_location='./model.pkl' ):

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.features = features
        self.target = target

        self.numerical_to_impute = numerical_to_impute
        self.numerical_to_scale = numerical_to_scale
        self.categorical_with_missing = categorical_with_missing
        self.categorical_with_rare = categorical_with_rare
        self.dict_of_freq_labels = dict_of_freq_labels
        self.dict_of_impute_values = {}
        self.final_feature_list = []

        self.test_size = test_size
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = LogisticRegression(self.random_state)

        self.save_train_flag = save_train_flag
        self.save_train_location = save_train_location
        self.save_test_flag = save_test_flag
        self.save_test_location = save_test_location
        self.save_model_flag = save_model_flag
        self.save_model_location = save_model_location

    '''Pre-processing functions:'''
    '''Learning Parameters from the training data'''
    def calculate_imputation_replacement(self):
        for var in self.numerical_to_impute:
            replacement = self.X_train[var].median()
            self.dict_of_impute_values[var] = replacement
        return self

    '''Functions on the data'''
    def add_missing_indicator(self, df):
        df = df.copy()
        for var in self.numerical_to_impute:
            df[var + '_NA'] = np.where(df[var].inull(), 1, 0)
        return df

    def impute_numerical_na(self, df):
        df = df.copy()
        for var in self.numerical_to_impute:
            df[var] = df[var].fillna(self.dict_of_impute_values[var])
        return df

    def replace_categorical_na(self, df):
        df = df.copy()
        for var in self.categorical_with_missing:
            df[var] = df[var].fillna("Missing")
        return df

    def remove_rare_labels(self, df):
        for var in self.categorical_with_rare:
            df[var] = np.where(df[var].isin(self.dict_of_freq_labels), df[var], "Rare")
        return df

    def encode_categorical(self, df):
        df = df.copy()
        df = pd.get_dummies(df, drop_first=True)
        return df

    def test_data_dummy_variable_check(self, df_to_score):
        df_to_score = df_to_score.copy()
        vars_to_add = set(self.features).difference(df_to_score.columns)
        for var in vars_to_add:
            df_to_score[var] = 0
        return df_to_score


    '''Master Functions'''
    def fit(self, data):
        '''Will fit the model INCLUDING a train/test split on the data.'''

        # train/test split:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.drop(self.target, axis=1),
            data[self.target],
            test_size=self.test_size,
            random_state=self.random_state)

        # calculating parameters:
        self.calculate_imputation_replacement()

        # add missing indicator:
        self.X_train = self.add_missing_indicator(self.X_train)
        self.X_test = self.add_missing_indicator(self.X_test)

        # impute missing numerical values:
        self.X_train = self.impute_numerical_na(self.X_train)
        self.X_test = self.impute_numerical_na(self.X_test)

        # fill in categorical na's with "Missing":
        self.X_train = self.replace_categorical_na(self.X_train)
        self.X_test = self.replace_categorical_na(self.X_test)

        # removing rare labels:
        self.X_train = self.remove_rare_labels(self.X_train)
        self.X_test = self.remove_rare_labels(self.X_test)

        # one-hot encoding of categorical variables:
        self.X_train = self.encode_categorical(self.X_train)
        self.X_test = self.encode_categorical(self.X_test)

        # add columns to the test set that may have not been captures through one-hot encoding:
        self.X_test = self.test_data_dummy_variable_check(self.X_test)

        # training the standard scaler:
        self.scaler.fit(self.X_train[self.numerical_to_scale])

        # scaling those variables:
        self.X_train[self.numerical_to_scale] = self.scaler.transform(self.X_train[self.numerical_to_scale])
        self.X_test[self.numerical_to_scale] = self.scaler.transform(self.X_test[self.numerical_to_scale])

        # making sure the variables are in the same order for training and testing:
        self.final_feature_list = self.X_train.columns
        self.X_test = self.X_test[self.final_feature_list]

        # saving, if save flags are True:
        if self.save_train_flag:
            self.X_train.to_csv(self.save_train_location, index=False)
        if self.save_test_flag:
            self.X_test.to_csv(self.save_test_location, index=False)

        # training the model:
        self.model.fit(self.X_train, self.y_train)
        if self.save_model_flag:
            joblib.dump(self.model, self.save_model_location)

        return self

    def _transform(self, data):
        '''Will transform a dataset to be ready for prediction.'''
        data = data.copy()

        # add missing indicator:
        data = self.add_missing_indicator(data)

        # impute missing numerical values:
        data = self.impute_numerical_na(data)

        # fill in categorical na's with "Missing":
        data = self.replace_categorical_na(data)

        # removing rare labels:
        data = self.remove_rare_labels(data)

        # one-hot encoding of categorical variables:
        data = self.encode_categorical(data)

        # add columns to the test set that may have not been captures through one-hot encoding:
        data = self.test_data_dummy_variable_check(data)

        # scaling variables:
        data[self.numerical_to_scale] = self.scaler.transform(data[self.numerical_to_scale])

        # making sure the variables are in the same order for training and testing:
        data = data[self.final_feature_list]

        return data

    def predict(self):
        '''Will predict on new data.'''
        return

    def evaluate_model(self):
        return



