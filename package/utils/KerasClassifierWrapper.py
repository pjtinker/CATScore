from sklearn.base import BaseEstimator, ClassifierMixin

class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y):
        self.model.get_params()['input_shape']= X['input_shape']
        self.model.fit(X['transformed'], y)
        return self
    
    def predict(self, X):
        return self.model.predict(X['transformed'])