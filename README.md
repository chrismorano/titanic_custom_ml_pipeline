# Titanic Survival Predictions
## Custom Pipeline

This project is part of the [Deployment of ML Models Course on Udemy](https://www.udemy.com/course/deployment-of-machine-learning-models/).  

Using the famous titanic dataset, this code using custom pipelines for ETL and model training to go from a download of the titanic data set to being able to make predictions with the current logistic regression model.

Feel free to fork the repo, try it for yourself, and play around with making the code better.  

Simply run the command ```python run_pipeline.py``` from the command line to get the accuracy and ROC-AUC scores for both the training and test sets.  By default, the data is downloaded, re-saved after the ETL process, and re-saved again in training and testing sets.  The model and the scaler are also saved automatically.
