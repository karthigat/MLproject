import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from tqdm import tqdm

def save_object(filepath, obj):
    """
        This function create a pickle file.

        Args:
            filepath and file object to be converted as pickle file

    """

    try:
        dir_path = os.path.dirname(filepath)

        os.makedirs(dir_path, exist_ok=True)

        with open(filepath, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)        
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    """
        This util function is to evaluate model. This function performs training and prediction 
        for various models.

        Args
            Takes X_train, X_test, y_train, y_test, and models
        
        Return
            Dict of models and it's r2 score
    """
    try:
        report = {}
        
        for i in tqdm(range(len(list(models)))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train) # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)