import numpy as np

class LossFunctions:

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(np.sum(y_true*np.log(y_pred), axis=0))

# Example
y_true = np.array([[1,0,0],[0,1,0]]).T
y_pred = np.array([[0.7,0.2,0.1],[0.1,0.8,0.1]]).T

print("Mean Squared Error:", LossFunctions.mse(y_true, y_pred))
print("Categorical Cross Entropy:", LossFunctions.categorical_cross_entropy(y_true, y_pred))
