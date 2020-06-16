# Titanic Survival Predictions
## Custom Pipeline

This project is part of the [Deployment of ML Models Course on Udemy](https://www.udemy.com/course/deployment-of-machine-learning-models/).  

Using the famous titanic dataset, this code uses custom pipelines for ETL and model training to go from a download of the titanic data to making predictions with a logistic regression model.

Feel free to fork the repo, try it for yourself, and play around with the code to make it better!  

You can create a conda environment with the package-list file (details are in the file itself, if needed).

Simply run the command ```python run_pipeline.py``` from the ```src/``` directory to get the accuracy and ROC-AUC scores for both the training and test sets.  By default, this command will download the data, re-save it after the ETL process, and save the training and testing sets after all feature engineering steps are applied.  The model and the scaler are also saved automatically in the models folder.  To change the defaults, just edit the config.yml file in the configs folder.
