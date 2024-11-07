import numpy as np

def sigmoid(pred):
    return 1/(1+np.exp(-pred))

class LogisticRegression:

    def __init__(self, lr = 0.001, num_iters = 500):
        self.lr = lr
        self.num_iters = num_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            prediction = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, prediction - y)
            db = (1/n_samples) * np.sum(prediction - y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self, X):
        lin_pred = np.dot(X, self.weights) + self.bias

        y_pred = sigmoid(lin_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred