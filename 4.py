
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(42)

W1 = np.random.rand(2,2)
b1 = np.random.rand(1,2)
W2 = np.random.rand(2,1)
b2 = np.random.rand(1,1)

lr = 0.5
epochs = 10000

for i in range(epochs):

    # Forward
    h = sigmoid(np.dot(X, W1) + b1)
    out = sigmoid(np.dot(h, W2) + b2)

    # Backward
    error = y - out
    d_out = error * sigmoid_derivative(out)
    d_h = d_out.dot(W2.T) * sigmoid_derivative(h)

    W2 += h.T.dot(d_out) * lr
    b2 += np.sum(d_out, axis=0) * lr
    W1 += X.T.dot(d_h) * lr
    b1 += np.sum(d_h, axis=0) * lr

    if i % 2000 == 0:
        print(f"Epoch {i} - Loss: {np.mean(error**2):.6f}")

print("\nFinal Output:")
print(out)

