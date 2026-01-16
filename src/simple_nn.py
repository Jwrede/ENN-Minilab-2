import numpy as np


class SimpleNeuralNetwork:
    """
    Single-layer neural network (no hidden layer)
    trained with gradient descent and sigmoid activation.
    """

    def __init__(self, lr=0.1, epochs=500):
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []

        # parameters (initialized in fit)
        self.W = None  # (D, K)
        self.b = None  # (K,)

    # -------------------------------------------------
    # Forward pass components
    # -------------------------------------------------
    def sigmoid(self, a):
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-a))

    def forward(self, X):
        """
        Forward pass: compute activations and predictions.
        """
        a = X @ self.W + self.b  # (N, K)
        y_hat = self.sigmoid(a)   # (N, K)
        return a, y_hat

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------
    def compute_loss(self, y_hat, y):
        """
        Mean Squared Error loss
        """
        return 0.5 * np.mean((y_hat - y) ** 2)

    # -------------------------------------------------
    # Backward pass (gradient computation)
    # -------------------------------------------------
    def backward(self, X, y, a, y_hat):
        """
        Computes gradients dW and db for one gradient step.

        X: (N, D)
        y: (N, K)
        a: (N, K)   (not strictly needed here, but kept for later extensions)
        y_hat: (N, K)

        returns:
          dW: (D, K)
          db: (K,)
        """
        N = X.shape[0]
        
        delta = (y_hat - y) * y_hat * (1.0 - y_hat)  # (N, K)

        dW = X.T @ delta / N  # (D, K)

        db = np.mean(delta, axis=0)  # (K,)

        return dW, db

    # -------------------------------------------------
    # Gradient descent step (EXPLICIT)
    # -------------------------------------------------
    def gradient_step(self, dW, db):
        """
        Performs one gradient descent update
        """
        self.W -= self.lr * dW
        self.b -= self.lr * db

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    def fit(self, X, y):
        """
        Train the model using full-batch gradient descent.

        X: (N, D) input data
        y: (N, K) one-hot labels
        """
        # Init the weight matrices
        self.W = 0.01 * np.random.randn(X.shape[1], y.shape[1])
        self.b = np.zeros(y.shape[1])

        self.loss_history = []

        # Realize gradient descent (iteratively)
        for _ in range(self.epochs):
            # Forward pass
            a, y_hat = self.forward(X)
            
            # Compute loss and store in history
            loss = self.compute_loss(y_hat, y)
            self.loss_history.append(loss)
            
            # Backward pass
            dW, db = self.backward(X, y, a, y_hat)
            
            # Gradient step
            self.gradient_step(dW, db)
        
        return self.loss_history

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    def predict(self, X):
        """
        Predict class labels (0..K-1).
        """
        _, y_hat = self.forward(X)
        return np.argmax(y_hat, axis=1)
    
    def predict_proba(self, X):
        """
        Predict probabilities per class (sigmoid outputs).
        """
        _, y_hat = self.forward(X)
        return y_hat