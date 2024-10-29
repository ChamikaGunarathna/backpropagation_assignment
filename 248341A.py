import numpy as np
import pandas as pd
import csv

# ReLU Activation function
def relu(x):
    return np.maximum(0, x).astype(np.float32)

# ReLU derivative
def relu_derivative(x):
    return np.where(x > 0, 1, 0).astype(np.float32)

# Softmax activation function
def softmax(x):
    x = x.astype(np.float32)
    exps = np.exp(x - np.max(x, axis=1, keepdims=True)).astype(np.float32)
    return (exps / np.sum(exps, axis=1, keepdims=True)).astype(np.float32)

# Cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    logp = -np.log(y_pred[range(n_samples), y_true.argmax(axis=1)]).astype(np.float32)
    loss = np.sum(logp).astype(np.float32) / np.float32(n_samples)
    return loss.astype(np.float32)

# Cross-entropy loss derivative
def cross_entropy_loss_derivative(y_true, y_pred):
    return (y_pred - y_true).astype(np.float32)

# One-hot encoding for labels
def one_hot_encode(y, num_classes=4):
    y = np.array(y).flatten()
    one_hot = np.zeros((y.size, num_classes), dtype=np.float32)
    one_hot[np.arange(y.size), y] = 1
    return one_hot

# Assign custom weights and return a neurat network instance
def assign_custom_weights(weights_df,biases_df):
    # Split weights and biases, rows are hard coded from the .csv files
    weights = {
        "w1": weights_df.iloc[:14, 1:101].values.astype(np.float32),
        "w2": weights_df.iloc[14:114, 1:41].values.astype(np.float32),
        "w3": weights_df.iloc[114:, 1:5].values.astype(np.float32),
    }

    biases = {
        "b1": biases_df.iloc[0, 1:101].values.astype(np.float32),
        "b2": biases_df.iloc[1, 1:41].values.astype(np.float32),
        "b3": biases_df.iloc[2, 1:5].values.astype(np.float32),
    }

    # Create an instance of the network and assigning the gives biases
    nn = NeuralNetwork()
    # assigning given weights
    nn.w1 = weights['w1']
    nn.w2 = weights['w2']
    nn.w3 = weights['w3']
    # assigning given biases
    nn.b1 = biases['b1'].reshape(1, -1).astype(np.float32)
    nn.b2 = biases['b2'].reshape(1, -1).astype(np.float32)
    nn.b3 = biases['b3'].reshape(1, -1).astype(np.float32)
    
    return nn

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size=14, hidden_size1=100, hidden_size2=40, output_size=4):
        # Initialize weights and biases
        self.w1 = np.random.randn(input_size, hidden_size1).astype(np.float32)
        self.b1 = np.zeros((1, hidden_size1), dtype=np.float32)
        self.w2 = np.random.randn(hidden_size1, hidden_size2).astype(np.float32)
        self.b2 = np.zeros((1, hidden_size2), dtype=np.float32)
        self.w3 = np.random.randn(hidden_size2, output_size).astype(np.float32)
        self.b3 = np.zeros((1, output_size), dtype=np.float32)

    # Forward propagation
    def forward(self, X):
        X = X.astype(np.float32)
        self.z1 = np.dot(X, self.w1).astype(np.float32) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2).astype(np.float32) + self.b2
        self.a2 = relu(self.z2)
        self.z3 = np.dot(self.a2, self.w3).astype(np.float32) + self.b3
        self.a3 = softmax(self.z3)
        return self.a3

    # Backpropagation
    def backward(self, X, y_true, y_pred, learning_rate):
        X, y_true, y_pred = X.astype(np.float32), y_true.astype(np.float32), y_pred.astype(np.float32)
        m = y_true.shape[0]

        self.dz3 = cross_entropy_loss_derivative(y_true, y_pred)
        self.dw3 = (np.dot(self.a2.T, self.dz3) / np.float32(m)).astype(np.float32)
        self.db3 = (np.sum(self.dz3, axis=0, keepdims=True) / np.float32(m)).astype(np.float32)

        self.dz2 = (np.dot(self.dz3, self.w3.T) * relu_derivative(self.z2)).astype(np.float32)
        self.dw2 = (np.dot(self.a1.T, self.dz2) / np.float32(m)).astype(np.float32)
        self.db2 = (np.sum(self.dz2, axis=0, keepdims=True) / np.float32(m)).astype(np.float32)

        self.dz1 = (np.dot(self.dz2, self.w2.T) * relu_derivative(self.z1)).astype(np.float32)
        self.dw1 = (np.dot(X.T, self.dz1) / np.float32(m)).astype(np.float32)
        self.db1 = (np.sum(self.dz1, axis=0, keepdims=True) / np.float32(m)).astype(np.float32)

        # Update weights and biases
        self.w3 -= learning_rate * self.dw3
        self.b3 -= learning_rate * self.db3
        self.w2 -= learning_rate * self.dw2
        self.b2 -= learning_rate * self.db2
        self.w1 -= learning_rate * self.dw1
        self.b1 -= learning_rate * self.db1

    # Step function
    def step(self, X, Y, learning_rate):
        learning_rate = np.float32(learning_rate)
        # Forward pass
        y_pred = self.forward(X)
        # Compute loss
        loss = cross_entropy_loss(Y, y_pred)
        # Backward pass
        self.backward(X, Y, y_pred, learning_rate)
        return loss
    
    # Write gradients to csv files
    def write_gradients_to_csv(self,w_name:str,b_name:str):
        '''w_name and b_name shoudl end with .csv
        '''
        # Save gradients for biases
        with open(b_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.db1.flatten())
            writer.writerow(self.db2.flatten())
            writer.writerow(self.db3.flatten())
        
        # Save the array as a single row in a CSV file
        with open(w_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for i in self.dw1:
                writer.writerow(i.flatten())
            for i in self.dw2:
                writer.writerow(i.flatten())
            for i in self.dw3:
                writer.writerow(i.flatten())
    
    def train(self,X_train,Y_train,iteration : int, learning_rate: float):
        costs = []
        for i in range(iteration):
            loss = self.step(
                X=X_train,
                Y=Y_train,
                learning_rate=learning_rate
                )
            print(f"For iteration {i+1} the loss is {loss}")
            costs.append(loss)
        return costs

'''
Task_1
'''

# datapoint.txt content
# This is the 14 dimension datapoint X:
X=[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]

# This is the 1 dimension label for X:
label = [3]

# Prepare the X and label to make sure they process in the network
x = np.array(X).reshape(1, 14)
y = one_hot_encode(np.array(label)).astype(np.float32)

'''
Testing for W0 and b0 as in the instructions
'''
# Load weights and biases with headers for each layer
weights_df = pd.read_csv('Task_1/a/W.csv', header=None)
biases_df = pd.read_csv('Task_1/a/b.csv', header=None)

# Running for a single step (learning rate do not matter for Task1)
nn = assign_custom_weights(weights_df,biases_df)
_ = nn.step(X=x,Y=y,learning_rate=0.01)

# Saving the results to csv files
# nn.write_gradients_to_csv(w_name="pred_dw.csv",b_name="pred_db.csv")

'''
Testing for W1 and b1 as in the instructions
'''
# Load weights and biases with headers for each layer
weights_df = pd.read_csv('Task_1/b/w-100-40-4.csv', header=None)
biases_df = pd.read_csv('Task_1/b/w-100-40-4.csv', header=None)

# Running for a single step (learning rate do not matter for Task1)
nn = assign_custom_weights(weights_df,biases_df)
_ = nn.step(X=x,Y=y,learning_rate=0.01)

# Saving the results to csv files
nn.write_gradients_to_csv(w_name="dw.csv",b_name="db.csv")

'''
Task_2
'''
# Train dataset
x_train = pd.read_csv('Task_2/x_train.csv',header=None).to_numpy(dtype=np.float32)
y_train = pd.read_csv('Task_2/y_train.csv',header=None).to_numpy()
y_train = one_hot_encode(y_train).astype(np.float32)
# Test dataset
x_test = pd.read_csv('Task_2/x_test.csv',header=None).to_numpy(dtype=np.float32)
y_test = pd.read_csv('Task_2/y_test.csv',header=None).to_numpy()
y_test = one_hot_encode(y_test).astype(np.float32)

nn = NeuralNetwork()
costs = nn.train(
    X_train=x_train,
    Y_train=y_train,
    iteration=1000,
    learning_rate=0.1,
    )