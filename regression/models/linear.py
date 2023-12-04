from feature_selector.lasso import LassoSelector
import pandas as pd
import numpy as np

class LinearRegression:
    selected_features = None
    data = None
    target = None
    learning_rate = 0.01
    num_iterations = 1000
    weights = None
    bias = None
    X: pd.DataFrame = None
    y: pd.DataFrame = None

    def __init__(self,data:pd.DataFrame, target):
        self.data = data
        self.target = target

    def feature_select(self,method)->None:
        if method == 'lasso':
            lasso = LassoSelector(data = self.data,target=self.target)
            self.selected_features = lasso.select()
        self.X = self.data[self.selected_features]
        self.y = self.data[self.target]
        


    def train_model(self)->None:
        # Initialize weights and bias
        self.weights = np.zeros(self.data[self.selected_features])
        self.bias = 0

        # Gradient Descent
        for _ in range(self.num_iterations):
            predictions = self.predict(self.X)
            errors = predictions - self.y

            # Update weights and bias
            self.weights -= self.learning_rate * (1 / self.X.shape[0]) * np.dot(self.X.T, errors)
            self.bias -= self.learning_rate * (1 / self.X.shape[0]) * np.sum(errors)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

        


