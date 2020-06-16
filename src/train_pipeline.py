import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

import joblib


class TrainingPipeline:

    def __init__(self, features, target, numerical_to_impute, numerical_to_scale, categorical_with_missing,
                 categorical_with_rare, dict_of_freq_labels, test_size=0.2, random_state=0,
                 save_train_test_flag=True, save_train_filename='../data/titanic_training_data.csv',
                 save_test_filename='../data/titanic_testing_data.csv',
                 save_scaler_filename='../models/scaler.pkl', save_model_filename='../models/model.pkl'):

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_full = None
        self.test_full = None

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
        self.model = LogisticRegression(solver='liblinear', random_state=self.random_state)

        self.save_train_test_flag = save_train_test_flag
        self.save_train_filename = save_train_filename
        self.save_test_filename = save_test_filename

        self.save_scaler_filename = save_scaler_filename
        self.save_model_filename = save_model_filename

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
            df[var + '_NA'] = np.where(df[var].isnull(), 1, 0)
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
            df[var] = np.where(df[var].isin(self.dict_of_freq_labels[var]), df[var], "Rare")
        return df

    def encode_categorical(self, df):
        df = df.copy()
        df = pd.get_dummies(df, drop_first=True)
        return df

    def test_data_dummy_variable_check(self, df_to_score):
        df_to_score = df_to_score.copy()
        vars_to_add = set(self.final_feature_list).difference(df_to_score.columns)
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
        self.final_feature_list = self.X_train.columns
        self.X_test = self.test_data_dummy_variable_check(self.X_test)

        # training the standard scaler:
        self.scaler.fit(self.X_train[self.numerical_to_scale])
        joblib.dump(self.scaler, self.save_scaler_filename)

        # scaling those variables:
        self.X_train[self.numerical_to_scale] = self.scaler.transform(self.X_train[self.numerical_to_scale])
        self.X_test[self.numerical_to_scale] = self.scaler.transform(self.X_test[self.numerical_to_scale])

        # making sure the variables are in the same order for training and testing:
        self.X_test = self.X_test[self.final_feature_list]

        # saving, if save flags are True:
        if self.save_train_test_flag:
            # adding the target to the training set:
            self.train_full = self.X_train.copy()
            self.train_full[self.target] = self.y_train
            self.train_full.to_csv(self.save_train_filename, index=False)

            # adding the target to the testing set:
            self.test_full = self.X_test.copy()
            self.test_full[self.target] = self.y_test
            self.test_full.to_csv(self.save_test_filename, index=False)

        # training the model:
        self.model.fit(self.X_train, self.y_train)
        joblib.dump(self.model, self.save_model_filename)

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

    def predict(self, data):
        '''Will predict on new data.'''
        data = self._transform(data)
        predictions = self.model.predict(data)
        
        return predictions

    def evaluate_model(self):
        if self.X_train is None:
            print("Model is not yet trained.")
        else:
            train_preds = self.model.predict(self.X_train)
            test_preds = self.model.predict(self.X_test)

            print('')
            print('--------TRAINING ERRORS----------')
            print(f'Accuracy Score: {accuracy_score(self.y_train, train_preds)}')
            print(f'ROC AUC Score: {roc_auc_score(self.y_train, train_preds)}')
            print('')
            print('--------TESTING ERRORS-----------')
            print(f'Accuracy Score: {accuracy_score(self.y_test, test_preds)}')
            print(f'ROC AUC Score: {roc_auc_score(self.y_test, test_preds)}')
            print('')
            
        return None