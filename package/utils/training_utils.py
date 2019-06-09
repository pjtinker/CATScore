

import pandas as pd 
import numpy as np

import os
import random
import importlib 

from sklearn.model_selection import train_test_split

def get_model(model_module, model_class, hyperparameters=None):
    try:
        module = importlib.import_module(model_module)
        req_class = getattr(module, model_class)
        return req_class(hyperparameters)
    except Exception as e:
        print("Exception occured while loading class.\n", e)