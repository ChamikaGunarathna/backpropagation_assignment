import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size=14, hidden_size1=100, hidden_size2=40, output_size=4):
        # Setting the seed for reproducibility
        np.random.seed(341)
        # Initializing weights and biases
        self.w1 = np.random.randn(input_size, hidden_size1).astype(np.float32)
        self.b1 = np.zeros((1, hidden_size1), dtype=np.float32)
        self.w2 = np.random.randn(hidden_size1, hidden_size2).astype(np.float32)
        self.b2 = np.zeros((1, hidden_size2), dtype=np.float32)
        self.w3 = np.random.randn(hidden_size2, output_size).astype(np.float32)
        self.b3 = np.zeros((1, output_size), dtype=np.float32)
        
    # Assign custom weights and return a neurat network instance
    def assign_custom_weights(self, weights_df:pd.DataFrame,biases_df:pd.DataFrame):
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

        # assigning given weights
        self.w1 = weights['w1']
        self.w2 = weights['w2']
        self.w3 = weights['w3']
        # assigning given biases
        self.b1 = biases['b1'].reshape(1, -1).astype(np.float32)
        self.b2 = biases['b2'].reshape(1, -1).astype(np.float32)
        self.b3 = biases['b3'].reshape(1, -1).astype(np.float32)
    
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

    # Backward propagation
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

        # Updating weights and biases
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
    
    # Writing gradients to csv files
    def write_gradients_to_csv(self,w_name:str,b_name:str):
        '''w_name and b_name shoudl end with .csv
        '''
        # Save gradients for biases
        with open(b_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.db1.flatten())
            writer.writerow(self.db2.flatten())
            writer.writerow(self.db3.flatten())
        
        # Save gradients for weights
        with open(w_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for i in self.dw1:
                writer.writerow(i.flatten())
            for i in self.dw2:
                writer.writerow(i.flatten())
            for i in self.dw3:
                writer.writerow(i.flatten())
    
    # Predict for testing
    def predict(self,X,Y):
        y_pred = self.forward(X=X)
        loss = cross_entropy_loss(Y, y_pred)
        return loss
    
    # Train the function for a given iteration and using a given learning rate
    def train(self,X_train,Y_train,X_test,Y_test,iteration : int, learning_rate: float):
        training_costs = []
        testing_costs = []
        for i in range(iteration):
            # training cost
            train_cost = self.step(
                X=X_train,
                Y=Y_train,
                learning_rate=learning_rate
                )
            # testing cost
            test_cost = self.predict(
                X=X_test,
                Y=Y_test
            )
            training_costs.append(train_cost)
            testing_costs.append(test_cost)
            print(f"For iteration {i+1} the training loss is {train_cost} and testing cost is {test_cost}")
        return training_costs, testing_costs

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
nn = NeuralNetwork()
nn.assign_custom_weights(
    weights_df=weights_df,
    biases_df=biases_df
    )
_ = nn.step(X=x,Y=y,learning_rate=0.01)

# Saving the results to csv files
# nn.write_gradients_to_csv(w_name="pred_dw.csv",b_name="pred_db.csv")

'''
Testing for W1 and b1 as in the instructions
'''
# Load weights and biases with headers for each layer
weights_df = pd.read_csv('Task_1/b/w-100-40-4.csv', header=None)
biases_df = pd.read_csv('Task_1/b/b-100-40-4.csv', header=None)

# Running for a single step (learning rate do not matter for Task1)
nn.assign_custom_weights(
    weights_df=weights_df,
    biases_df=biases_df
    )
_ = nn.step(X=x,Y=y,learning_rate=0.01)

# Saving the results to csv files
nn.write_gradients_to_csv(w_name="dw.csv",b_name="db.csv")

'''
Task_2
'''
# exucuting as per given learning rates
isTrain = False
if isTrain:
    # Train dataset
    x_train = pd.read_csv('Task_2/x_train.csv',header=None).to_numpy(dtype=np.float32)
    y_train = pd.read_csv('Task_2/y_train.csv',header=None).to_numpy()
    y_train = one_hot_encode(y_train).astype(np.float32)
    # Test dataset
    x_test = pd.read_csv('Task_2/x_test.csv',header=None).to_numpy(dtype=np.float32)
    y_test = pd.read_csv('Task_2/y_test.csv',header=None).to_numpy()
    y_test = one_hot_encode(y_test).astype(np.float32)
    
    iterations = 10000
    learning_rates = [1,0.1,0.001]
    train_costs_list = []
    test_costs_list = []
    for learning_rate in learning_rates:
        nn = NeuralNetwork()
        train_costs, test_costs = nn.train(
            X_train=x_train,
            Y_train=y_train,
            X_test=x_test,
            Y_test=y_test,
            iteration=iterations,
            learning_rate=learning_rate,
            )
        train_costs_list.append(train_costs)
        test_costs_list.append(test_costs)

    # saving the results to a excel file
    iterations = list(range(1,iterations+1))
    df = pd.DataFrame({
        'iteration': iterations,
        'lr_1_train': train_costs_list[0],
        'lr_0.1_train': train_costs_list[1],
        'lr_0.001_train': train_costs_list[2],
        'lr_1_test': test_costs_list[0],
        'lr_0.1_test': test_costs_list[1],
        'lr_0.001_test': test_costs_list[2],
        })
    df.to_csv('cost_results.csv', index=False)

# plot charts (when df is created)
isPlot = True
if isPlot:
    df = pd.read_csv('cost_results.csv')
    # plotting for training costs vs iterations
    plt.figure(figsize=(10, 6))
    plt.scatter(df['iteration'], df['lr_1_train'], label='learning rate : 1', color='b',s=10)
    plt.scatter(df['iteration'], df['lr_0.1_train'], label='learning rate : 0.1', color='g',s=10)
    plt.scatter(df['iteration'], df['lr_0.001_train'], label='learning rate : 0.001', color='r',s=10)

    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Training Costs vs Iterations')
    plt.legend()
    plt.show()

    # plotting for testing costs vs iterations
    plt.figure(figsize=(10, 6))
    plt.scatter(df['iteration'], df['lr_1_test'], label='learning rate : 1', color='b',s=10)
    plt.scatter(df['iteration'], df['lr_0.1_test'], label='learning rate : 0.1', color='g',s=10)
    plt.scatter(df['iteration'], df['lr_0.001_test'], label='learning rate : 0.001', color='r',s=10)

    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Testing Costs vs Iterations')
    plt.legend()
    plt.show()
    
    # plotting for testing & training costs vs iterations for learning rate 1 
    plt.figure(figsize=(10, 6))
    plt.scatter(df['iteration'], df['lr_1_train'], label='training costs', color='r',s=10)
    plt.scatter(df['iteration'], df['lr_1_test'], label='testing costs', color='g',s=10)

    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Training and Testing Costs vs Iterations for Learning Rate 1')
    plt.legend()
    plt.show()
    
    # plotting for testing & training costs vs iterations for learning rate 0.1
    plt.figure(figsize=(10, 6))
    plt.scatter(df['iteration'], df['lr_0.1_train'], label='training costs', color='r',s=10)
    plt.scatter(df['iteration'], df['lr_0.1_test'], label='testing costs', color='g',s=10)

    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Training and Testing Costs vs Iterations for Learning Rate 0.1')
    plt.legend()
    plt.show()
    
    # plotting for testing & training costs vs iterations for learning rate 0.001
    plt.figure(figsize=(10, 6))
    plt.scatter(df['iteration'], df['lr_0.001_train'], label='training costs', color='r',s=10)
    plt.scatter(df['iteration'], df['lr_0.001_test'], label='testing costs', color='g',s=10)

    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Training and Testing Costs vs Iterations for Learning Rate 0.001')
    plt.legend()
    plt.show()