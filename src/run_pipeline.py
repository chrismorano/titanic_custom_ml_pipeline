import pandas as pd
import yaml

from train_pipeline import TrainingPipeline
from etl_pipeline import ETLPipeline
with open("../configs/config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

# Calling the pipelines:

etl_pipeline = ETLPipeline(features=cfg['FEATURES'],
                           target=cfg['TARGET'],
                           save_flag=True,
                           save_location=cfg['DATA_LOCATION_POST_ETL'],
                           save_filename='titanic')

training_pipeline = TrainingPipeline(features=cfg['FEATURES'],
                                     target=cfg['TARGET'],
                                     numerical_to_float=cfg['NUMERICAL_TO_FLOAT'],
                                     numerical_to_impute=cfg['NUMERICAL_TO_IMPUTE'],
                                     numerical_to_scale=cfg['NUMERICAL_TO_SCALE'],
                                     categorical_with_missing=cfg['CATEGORICAL_WITH_MISSING_VALUES'],
                                     categorical_with_rare=cfg['CATEGORICAL_WITH_RARE_LABELS'],
                                     dict_of_freq_labels=cfg['DICT_OF_FREQUENT_LABELS'])

if __name__ == '__main__':

    data = pd.read_csv(cfg['PATH_TO_DATA'])
    cleaning_pipeline.fit(data)

    # ------------------------

    # Model Evaluation:
    print("Model Performance:")
    pipeline.evaluate_model()