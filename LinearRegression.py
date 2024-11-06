import numpy as np

class LinearRegression:

    def __init__(self, lr = 0.001, num_iters = 1000):
        self.lr = lr
        self.num_iters = num_iters

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = None
        self.bias = None

        for i in range(self.num_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, y_pred - y)
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db
        

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias

        return y_pred