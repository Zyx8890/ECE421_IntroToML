import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 


def fit_NeuralNetwork(X_train,y_train,alpha,hidden_layer_sizes,epochs):
    """
    Train a neural network using gradient descent.
    
    Args:
        X_train: Training data (N x d)
        y_train: Training labels (N,)
        alpha: Learning rate
        hidden_layer_sizes: List of hidden layer sizes [h1, h2, ...]
        epochs: Number of training epochs
    
    Returns:
        errors: List of total errors per epoch
        weights: Final trained weights
    """
    N, d = X_train.shape
    
    # Initialize network architecture
    layer_sizes = [d] + hidden_layer_sizes + [1]  # [input, hidden1, hidden2, ..., output]
    L = len(layer_sizes)  # Total number of layers
    
    # Initialize weights with He initialization for ReLU
    weights = []
    for l in range(L - 1):
        # weights[l] has shape (d^l + 1) x d^(l+1)
        # +1 for bias in the input layer
        # He initialization: scale by sqrt(2 / fan_in)
        fan_in = layer_sizes[l]
        w = np.random.randn(layer_sizes[l] + 1, layer_sizes[l + 1]) * np.sqrt(2.0 / fan_in)
        weights.append(w)
    
    # Training loop
    errors = []
    for epoch in range(epochs):
        epoch_error = 0
        
        for n in range(N):
            # Prepare input with bias
            x_n = np.concatenate([[1], X_train[n]])  # Add bias (prepend 1)
            y_n = y_train[n]
            
            # Forward propagation
            X, S = forwardPropagation(x_n, weights)
            
            # Compute error for this sample
            error = errorPerSample(X, y_n)
            epoch_error += error
            
            # Backpropagation
            gradients = backPropagation(X, y_n, S, weights)
            
            # Update weights
            weights = updateWeights(weights, gradients, alpha)
        
        errors.append(epoch_error)
    
    return errors, weights
    
def forwardPropagation(x, weights):
    """
    Forward propagation through the neural network.
    
    Args:
        x: Input vector of size d+1 (first element is 1 for bias)
        weights: List of weight matrices. weights[l] has shape (d^l + 1) x d^(l+1)
    
    Returns:
        X: List of outputs at all nodes in all layers
           X[l] is a vector of size d^l + 1 (except last layer which is single value)
        S: List of inputs to nodes before activation (L-1 elements)
           S[l] is a vector of size d^(l+1)
    """
    L = len(weights) + 1  # Total number of layers (input + hidden + output)
    X = []  # Store outputs at each layer
    S = []  # Store inputs before activation at each layer
    
    # Layer 0: Input layer
    X.append(np.array(x))  # X[0] = input x
    
    # Layers 1 to L-1: Hidden and output layers
    for l in range(len(weights)):
        # Compute s_j^l = sum(w_ij^(l-1) * x_i^(l-1))
        # This is matrix multiplication: weights[l].T @ X[l]
        s = weights[l].T @ X[l]
        S.append(s)
        
        # Apply activation function
        if l < len(weights) - 1:
            # Hidden layers: use ReLU activation and add bias
            x_next = np.array([activation(s_j) for s_j in s])
            # Add bias node (prepend 1)
            x_next = np.concatenate([[1], x_next])
        else:
            # Output layer: use output function (sigmoid), no bias added
            x_next = outputf(s[0])  # Single output node
        
        X.append(x_next)
    
    return X, S

def errorPerSample(X,y_n):
    """
    Compute the error for a single sample.
    
    Args:
        X: List of outputs at all layers (from forwardPropagation)
        y_n: True label for this sample
    
    Returns:
        error: The error value
    """
    # X[-1] is the output of the network
    x_L = X[-1]
    return errorf(x_L, y_n)

def backPropagation(X,y_n,s,weights):
    """
    Backpropagation to compute gradients.
    
    Args:
        X: List of outputs at all layers (from forwardPropagation)
        y_n: True label for this sample
        s: List S of pre-activation values (from forwardPropagation)
        weights: List of weight matrices
    
    Returns:
        gradients: List of gradient matrices, same structure as weights
    """
    L = len(weights) + 1  # Total number of layers
    deltas = [None] * len(weights)  # Store delta for each layer
    gradients = []  # Store gradients for each weight matrix
    
    # Output layer: delta^L = dE/dx_L * dx_L/ds_L
    x_L = X[-1]  # Output of the network
    delta_L = derivativeError(x_L, y_n) * derivativeOutput(s[-1][0])
    deltas[-1] = np.array([delta_L])
    
    # Backpropagate through hidden layers
    for l in range(len(weights) - 2, -1, -1):
        # delta^l = (weights^l @ delta^(l+1)) * activation'(s^l)
        # Remove the bias term from delta propagation
        delta_next = deltas[l + 1]
        delta = weights[l + 1][1:, :] @ delta_next  # Skip bias weight
        # Apply derivative of activation function element-wise
        delta = delta * np.array([derivativeActivation(s_j) for s_j in s[l]])
        deltas[l] = delta
    
    # Compute gradients: g_ij^l = x_i^l * delta_j^(l+1)
    for l in range(len(weights)):
        # Outer product: X[l] is column vector, deltas[l] is column vector
        # gradient = X[l][:, None] @ deltas[l][None, :]
        gradient = np.outer(X[l], deltas[l])
        gradients.append(gradient)
    
    return gradients

def updateWeights(weights,g,alpha):
    """
    Update weights using gradient descent.
    
    Args:
        weights: List of current weight matrices
        g: List of gradient matrices (from backPropagation)
        alpha: Learning rate
    
    Returns:
        updated_weights: List of updated weight matrices
    """
    updated_weights = []
    for l in range(len(weights)):
        # Gradient descent: w_new = w_old - alpha * gradient
        w_new = weights[l] - alpha * g[l]
        updated_weights.append(w_new)
    return updated_weights

def activation(s):
    #Enter implementation here
    '''We are implementing a ReLU activation function here
    The output of the ReLU function is 0 if the input is less than 0, 
    and remains the same if input is greater than or equal to 0
    '''
    if s>=0:
        return s
    else:
        return 0

def derivativeActivation(s):
    #Enter implementation here
    '''We are implementing the derivative of the ReLU activation function here
    The derivative of the ReLU function is 0 if the input is less than 0, 
    and 1 if the input is greater than or equal to 0
    '''
    if s>=0:
        return 1
    else:
        return 0

def outputf(s):
    #Enter implementation here
    '''
    We are implementing a logistic regression output function like sigmoid function
    Converting the input to a value between 0 and 1, which can be interpreted as a probability,
    and subsequently used to calculate the error and update the weights during backpropagation.
    '''
    return 1/(1+np.exp(-s))

def derivativeOutput(s):
    #Enter implementation here
    '''We are implementing the derivative of the logistic regression output function like sigmoid function
    The derivative of the sigmoid function is sigmoid(s) * (1 - sigmoid(s))
    '''
    return outputf(s) * (1 - outputf(s))

def errorf(x_L,y):
    #Enter implementation here
    '''
    This function calculates the error for a single sample based on the output of the network (x_L) and the true label (y).
    We are using the binary cross-entropy loss function, which is commonly used for binary classification problems. The error is calculated as:
    - If the true label y is 1, the error is -log(x_L), where x_L is the output of the network (the predicted probability of the positive class).
    - If the true label y is 0, the error is -log(1 - x_L), where (1 - x_L) is the predicted probability of the negative class.
    '''
    # Clip to avoid log(0) which gives inf/nan
    x_L = np.clip(x_L, 1e-15, 1 - 1e-15)
    if y==1:
        return -np.log(x_L)
    else:
        return -np.log(1-x_L)
    
def derivativeError(x_L,y):
    #Enter implementation here
    # This function finds the derivative of the error calculated in errorf
    # Clip to avoid division by zero
    x_L = np.clip(x_L, 1e-15, 1 - 1e-15)
    if y==1:
        return -1/x_L
    else:
        return 1/(1-x_L)

   
def pred(x_n,weights):
    # Forward propagate input x_n through the network
    x_n = np.concatenate([[1], x_n])  # Add bias
    X, _ = forwardPropagation(x_n, weights)
    output = X[-1]
    # Threshold at 0.5 for binary classification
    return 1 if output >= 0.5 else -1

def confMatrix(X_train,y_train,w):
    # Predict for each sample and build confusion matrix
    y_pred = []
    for i in range(len(X_train)):
        y_hat = pred(X_train[i], w)
        y_pred.append(y_hat)
    # Convert labels to 0/1 for confusion matrix
    y_true = [1 if y == 1 else 0 for y in y_train]
    y_pred_bin = [1 if y == 1 else 0 for y in y_pred]
    return confusion_matrix(y_true, y_pred_bin)

def plotErr(e,epochs):
    # Plot error vs epochs
    plt.figure()
    plt.plot(range(epochs), e)
    plt.xlabel('Epochs')
    plt.ylabel('Total Error')
    plt.title('Error vs Epochs')
    plt.show()
    
def test_SciKit(X_train, X_test, Y_train, Y_test):
    # Use scikit-learn MLPClassifier for comparison
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(8,), random_state=1, max_iter=100)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    # Convert labels to 0/1 for confusion matrix
    y_true = [1 if y == 1 else 0 for y in Y_test]
    y_pred_bin = [1 if y == 1 else 0 for y in y_pred]
    return confusion_matrix(y_true, y_pred_bin)

def test():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2, random_state=1)
    
    # Standardize features (zero mean, unit variance)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    
    for i in range(80):
        if y_train[i]==1:
            y_train[i]=-1
        else:
            y_train[i]=1
    for j in range(20):
        if y_test[j]==1:
            y_test[j]=-1
        else:
            y_test[j]=1
        
    err,w=fit_NeuralNetwork(X_train,y_train,0.1,[8],100)
    
    plotErr(err, 100)
    
    cM=confMatrix(X_test,y_test,w)
    
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)

test()