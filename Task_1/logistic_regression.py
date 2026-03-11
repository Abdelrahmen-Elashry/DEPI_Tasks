import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class LogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=0.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.cost_history = []

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _compute_cost(self, y, y_pred, m):
        cost = -(1 / m) * np.sum(
            y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        )
        return cost

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1)

        m, n = X.shape

        self.weights = np.zeros(n)
        self.bias = 0
        self.cost_history = []

        for i in range(self.n_iterations):
            z = X.dot(self.weights) + self.bias
            y_pred = self._sigmoid(z)

            error = y_pred - y

            dw = (1 / m) * X.T.dot(error)
            db = (1 / m) * np.sum(error)

            if self.regularization > 0:
                dw += (self.regularization / m) * self.weights

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            cost = self._compute_cost(y, y_pred, m)
            self.cost_history.append(cost)

        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        z = X.dot(self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def score(self, X, y):
        y = np.array(y, dtype=np.float64).reshape(-1)
        predictions = self.predict(X)
        return np.mean(predictions == y)