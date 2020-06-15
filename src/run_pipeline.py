import pandas as pd
import yaml

from etl_pipeline import ETLPipeline
from train_pipeline import TrainingPipeline
with open("../configs/config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

# Calling the pipelines:

etl_pipeline = ETLPipeline(features=cfg['FEATURES'],
                           target=cfg['TARGET'],
                           numerical_to_float=cfg['NUMERICAL_TO_FLOAT'],
                           save_filename=cfg['TITANIC_POST_ETL'])

training_pipeline = TrainingPipeline(features=cfg['FEATURES'],
                                     target=cfg['TARGET'],
                                     numerical_to_impute=cfg['NUMERICAL_TO_IMPUTE'],
                                     numerical_to_scale=cfg['NUMERICAL_TO_SCALE'],
                                     categorical_with_missing=cfg['CATEGORICAL_WITH_MISSING_VALUES'],
                                     categorical_with_rare=cfg['CATEGORICAL_WITH_RARE_LABELS'],
                                     dict_of_freq_labels=cfg['DICT_OF_FREQUENT_LABELS'],
                                     save_train_filename=cfg['TITANIC_X_TRAIN'],
                                     save_test_filename=cfg['TITANIC_X_TEST'],
                                     save_model_filename=cfg['TITANIC_MODEL'])

if __name__ == '__main__':

    data = pd.read_csv(cfg['TITANIC_DATA'])
    data = etl_pipeline.transform(data)
    training_pipeline.fit(data)

    # ------------------------

    # Model Evaluation:
    training_pipeline.evaluate_model()
