import numpy as np

class Activation:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def leaky_relu(x):
        return np.where(x > 0, x, 0.01*x)

    @staticmethod
    def elu(x):
        return np.where(x > 0, x, np.exp(x)-1)

    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

# Example
x = np.array([-2, -1, 0, 1, 2])

print("ReLU:", Activation.relu(x))
print("Sigmoid:", Activation.sigmoid(x))
print("Tanh:", Activation.tanh(x))
print("Leaky ReLU:", Activation.leaky_relu(x))
print("ELU:", Activation.elu(x))
print("Softmax:", Activation.softmax(x))
