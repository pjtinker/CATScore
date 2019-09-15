import pandas as pd 
import numpy as np 

from sklearn.base import BaseEstimator, TransformerMixin

from tensorflow.python.keras.preprocessing import sequence, text

MAX_SEQUENCE_LENGTH = 1500

class SequenceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, tokenizer, max_sequence_len):
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len

    def fit(self, X, y=None):
        # self.tokenizer.fit_on_texts([str(word) for word in X])
        return self

    def transform(self, X):
        res = self.tokenizer.texts_to_sequences([str(word) for word in X])
        max_len = len(max(X, key=len))
        if max_len > self.max_sequence_len:
            max_len = self.max_sequence_len

        res = sequence.pad_sequences(res, maxlen=max_len)
        global INPUT_SHAPE 
        self.input_length = res.shape[1:]
        INPUT_SHAPE = res.shape[1:]
        print(f"********** INPUT_SHAPE assigned by SequenceTransformer: {INPUT_SHAPE}")
        return {'transformed' : res, 'input_shape' : res.shape[1:]}
        # return res
    
    # def get_input_shape(self):
    #     return self.input_length