import json
import os
import inspect
import importlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from tinydb import TinyDB

my_module = importlib.import_module('sklearn.feature_extraction.text')
my_class = getattr(my_module, 'CountVectorizer')
cv = my_class()
print("Module name: {}".format(type(cv).__name__))
models = [MultinomialNB(), RandomForestClassifier(), SVC(), XGBClassifier()]
params = {}
tfidf = TfidfVectorizer()
tfidf_params = {type(tfidf).__name__ + '__' + k:v for k,v in tfidf.get_params().items() if not inspect.isclass(v)}

def dumper(obj):
    try: 
        return obj.toJSON()
    except:
        return
   
v0 = TinyDB('v0.json')
if len(v0.tables()) > 0:
    v0.purge_tables()
for model in models:
    table = v0.table(type(model).__name__)
    params = {}
    params['model_params'] = {}
    params['model_params'] = {type(model).__name__ + '__' + k:v for k,v in model.get_params().items() if not inspect.isclass(v)}
    params['tfidf_params'] = {}
    params['tfidf_params'] = tfidf_params
    params['question_number'] = 'default'
    table.insert(params)
    # table.insert({'doc' : model.__doc__})


    