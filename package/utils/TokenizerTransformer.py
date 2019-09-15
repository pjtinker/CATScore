from keras.preprocessing.text import Tokenizer
from sklearn.base import BaseEstimator, TransformerMixin


class TokenizerTransformer(BaseEstimator, TransformerMixin, Tokenizer):

    def __init__(self, tokenizer, **tokenizer_params):
        # Tokenizer.__init__(self, **tokenizer_params)
        self.tokenizer = tokenizer
        
    def fit(self, X, y=None):
        # self.tokenizer.fit_on_texts(X)
        return self

    def transform(self, X, y=None):
        X_transformed = self.tokenizer.texts_to_sequences(X)
        return X_transformed
    
    def get_num_features(self):
        print("get_num_features called from TokenizerTransformer")
        print(f"num_features: {len(self.tokenizer.word_index)}")
        return min(len(self.tokenizer.word_index) + 1, 20000)