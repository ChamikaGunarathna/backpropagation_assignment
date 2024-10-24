import numpy as np

# reLU Activation function
def relu(x):
    return np.maximum(0, x)

# reLU derivative
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Softmax activation function
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    logp = - np.log(y_pred[range(n_samples), y_true.argmax(axis=1)])
    loss = np.sum(logp) / n_samples
    return loss

# Cross-entropy loss derivative
def cross_entropy_loss_derivative(y_true, y_pred):
    return y_pred - y_true

# One-hot encoding for labels
def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size=14, hidden_size1=100, hidden_size2=40, output_size=4):
        # Initialize weights and biases
        self.w1 = np.random.randn(input_size, hidden_size1)
        self.b1 = np.zeros((1, hidden_size1))
        self.w2 = np.random.randn(hidden_size1, hidden_size2)
        self.b2 = np.zeros((1, hidden_size2))
        self.w3 = np.random.randn(hidden_size2, output_size)
        self.b3 = np.zeros((1, output_size))

    # Forward propagation
    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = relu(self.z2)
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(self.z3)
        return self.a3

    # Backpropagation
    def backward(self, X, y_true, y_pred, learning_rate):
        m = y_true.shape[0]
        
        self.dz3 = cross_entropy_loss_derivative(y_true, y_pred)
        self.dw3 = np.dot(self.a2.T, self.dz3) / m
        self.db3 = np.sum(self.dz3, axis=0, keepdims=True) / m

        self.dz2 = np.dot(self.dz3, self.w3.T) * relu_derivative(self.z2)
        self.dw2 = np.dot(self.a1.T, self.dz2) / m
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True) / m

        self.dz1 = np.dot(self.dz2, self.w2.T) * relu_derivative(self.z1)
        self.dw1 = np.dot(X.T, self.dz1) / m
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.w3 -= learning_rate * self.dw3
        self.b3 -= learning_rate * self.db3
        self.w2 -= learning_rate * self.dw2
        self.b2 -= learning_rate * self.db2
        self.w1 -= learning_rate * self.dw1
        self.b1 -= learning_rate * self.db1
        
    #instead of train, use a step function
    def step(self, X, Y, learning_rate):
        # Forward pass
        y_pred = self.forward(X)
        # Compute loss
        loss = cross_entropy_loss(Y, y_pred)
        # Backward pass
        self.backward(X, Y, y_pred, learning_rate)

def main():
    # Sample data
    np.random.seed(0)
    X_train = np.random.randn(100, 14)  # 100 samples, 14 features
    y_train = np.random.randint(0, 4, size=100)  # 100 labels for 4 classes
    y_train = one_hot_encode(y_train, 4)

    # Initialize and train the neural network
    nn = NeuralNetwork()
    nn.train(X_train, y_train, learning_rate=0.1, iterations=1000)

if __name__ == "__main__":
    main()