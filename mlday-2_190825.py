import numpy as np

# Data (XOR truth table)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(2, 2)   # input -> hidden
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)   # hidden -> output
b2 = np.zeros((1, 1))

# Sigmoid + derivative
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)

# Training
lr = 0.1
for epoch in range(10000):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)
    
    # Loss (MSE for simplicity)
    loss = np.mean((y - y_pred)**2)

    # Backprop
    d_loss = (y_pred - y)
    d_z2 = d_loss * sigmoid_deriv(y_pred)
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * sigmoid_deriv(a1)
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    # Update weights
    W1 -= lr * d_W1
    b1 -= lr * d_b1
    W2 -= lr * d_W2
    b2 -= lr * d_b2

    # Print progress every 2000 epochs
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final predictions
print("\nPredictions after training:")
print(y_pred.round(3))
