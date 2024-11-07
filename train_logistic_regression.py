import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt


breast_cancer_data = datasets.load_breast_cancer()
X, y = breast_cancer_data.data, breast_cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 124)

clf = LogisticRegression()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

def accuracy(prediction, y_test):
    return np.sum(prediction == y_test)/ len(y_test)

acc = accuracy(prediction, y_test)

print(acc)

