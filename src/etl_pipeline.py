import re
import numpy as np
# import pandas as pd


class ETLPipeline:

    def __init__(self, features, target, numerical_to_float, save_raw_flag=True,
                 save_raw_filename='../data/titanic_raw.csv', save_clean_data_flag=True,
                 save_clean_data_filename='../data/titanic_post_etl.csv'):

        self.features = features
        self.target = target
        self.numerical_to_float = numerical_to_float

        self.save_raw_flag = save_raw_flag
        self.save_raw_filename = save_raw_filename
        self.save_clean_data_flag = save_clean_data_flag
        self.save_clean_data_filename = save_clean_data_filename
    

    def basic_data_cleaning(self, df):
        '''Keeps only relevant columns, replacing "?" with NaN, and correcting data types'''
        df = df.copy()
        df = df.replace('?', np.nan)
        df.drop(labels=['ticket', 'boat', 'body', 'home.dest'], axis=1, inplace=True)
        return df

    def cast_to_float(self, df):
        ''' '''
        df = df.copy()
        for var in self.numerical_to_float:
            df[var] = df[var].astype('float')
        return df

    @staticmethod
    def get_title(line):
        ''' '''
        if re.search('Mrs', line):
            return 'Mrs'
        elif re.search('Mr', line):
            return 'Mr'
        elif re.search('Miss', line):
            return 'Miss'
        elif re.search('Master', line):
            return 'Master'
        else:
            return 'Other'

    def replace_name_with_title(self, df):
        ''' '''
        df = df.copy()
        df['title'] = df['name'].apply(self.get_title)
        df.drop(labels=['name'], axis=1, inplace=True)

        return df

    def extract_cabin_letter(self, df):
        ''' '''
        df = df.copy()
        df['cabin'] = df['cabin'].str.replace(r'\d+', '').str[0]
        return df

    # ------------------------------

    def transform(self, data):
        ''' '''
        data = data.copy()
        if self.save_raw_flag:
            data.to_csv(self.save_raw_filename, index=False)
        data = self.basic_data_cleaning(data)
        data = self.cast_to_float(data)
        data = self.replace_name_with_title(data)
        data = self.extract_cabin_letter(data)
        data = data[self.features + [self.target]]
        if self.save_clean_data_flag:
            data.to_csv(self.save_clean_data_filename, index=False)

        return data