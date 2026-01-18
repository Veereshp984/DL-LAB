import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(a):
    return a*(1-a)

class NeuralNet:
    def __init__(self, lr=0.1):
        np.random.seed(42)
        self.W1 = np.random.randn(4,2)
        self.b1 = np.zeros((4,1))
        self.W2 = np.random.randn(1,4)
        self.b2 = np.zeros((1,1))
        self.lr = lr

    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = np.maximum(0, self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, X, Y):
        m = Y.shape[1]

        dZ2 = (self.A2 - Y) * sigmoid_deriv(self.A2)
        dW2 = np.dot(dZ2, self.A1.T)/m
        dB2 = np.sum(dZ2, axis=1, keepdims=True)/m

        dZ1 = np.dot(self.W2.T, dZ2) * (self.A1 > 0)
        dW1 = np.dot(dZ1, X.T)/m
        dB1 = np.sum(dZ1, axis=1, keepdims=True)/m

        self.W1 -= self.lr*dW1
        self.b1 -= self.lr*dB1
        self.W2 -= self.lr*dW2
        self.b2 -= self.lr*dB2

    def train(self, X, Y, epochs):
        for i in range(epochs+1):
            out = self.forward(X)
            loss = np.mean((Y-out)**2)
            self.backward(X,Y)

            if i % 200 == 0:
                print(f"Epoch {i:4d} | Loss: {loss:.5f}")

X = np.array([[0,0,1,1],
              [0,1,0,1]])

Y = np.array([[0,1,1,0]])

nn = NeuralNet()
nn.train(X, Y, 4000)

print("\nPredictions:")
print(nn.forward(X).round(3))
