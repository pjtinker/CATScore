import pandas as pd 
import numpy as np 

from sklearn.base import BaseEstimator, TransformerMixin

from tensorflow.python.keras.preprocessing import sequence, text

class SequenceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, num_words=20000):
        self.num_words = num_words 
        self.tokenizer = text.Tokenizer(num_words=self.num_words)

    def fit(self, X, y=None):
        self.tokenizer.fit_on_texts([str(word) for word in X])
        return self

    def transform(self, X):
        res = self.tokenizer.texts_to_sequences([str(word) for word in X])
        return res