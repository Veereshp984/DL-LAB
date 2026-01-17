import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.init_params()

    def init_params(self):
        np.random.seed(42)

        for i in range(len(self.layers) - 1):
            w = np.random.randn(self.layers[i+1], self.layers[i])
            b = np.zeros((self.layers[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)

    def summary(self):
        print("Neural Network Architecture:")
        for i in range(len(self.weights)):
            print(f" Layer {i} -> Layer {i+1}: "
                  f"Weights {self.weights[i].shape}, "
                  f"Biases {self.biases[i].shape}")

# Example
nn = NeuralNetwork([3, 5, 2])
nn.summary()