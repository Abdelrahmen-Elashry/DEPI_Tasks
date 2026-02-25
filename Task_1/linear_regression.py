import numpy as np
import random
from sklearn.base import BaseEstimator, RegressorMixin

class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.w = np.zeros(X.shape[1])
        self.b = 0

        for epoch in range(self.epochs):
            for i in range(len(X)):
                point_x = X[i]
                point_y = y[i]

                prediction = np.dot(point_x, self.w) + self.b

                error = point_y - prediction

                self.w += self.learning_rate * error * point_x
                self.b += self.learning_rate * error
                
        return self

    def predict(self, X):
        return np.dot(X, self.w) + self.b