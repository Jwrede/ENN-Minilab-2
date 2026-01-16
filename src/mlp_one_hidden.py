import numpy as np

class MLPOneHiddenLayer:
    """
    MLP with exactly one hidden layer and sigmoid activations.
    - Output uses sigmoid per class (one-hot labels expected).
    - Loss: MSE (same style as the simple_nn baseline).
    """

    def __init__(self, hidden_dim=5, lr=0.01, epochs=1000, seed=42):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.loss_history = []

        self.input_dim = 2
        self.output_dim = 3

        # parameters (set in _init_params)
        # Note: This is realized as 
        # - input dimensions in first dimension
        # - hidden layer in second dimension
        # in an np.array.
        # Leads to computation order: Input array multiplied by matrix.
        self.W1 = None  # (D, H)
        self.b1 = None  # (H,)
        self.W2 = None  # (H, K)
        self.b2 = None  # (K,)

    # -------------------------
    # Init function
    # -------------------------
    def reset_weights(self):
        """
        Initialize weights and biases.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.W1 = 0.01 * np.random.randn(self.input_dim, self.hidden_dim)
        self.b1 = np.zeros(self.hidden_dim)

        self.W2 = 0.01 * np.random.randn(self.hidden_dim, self.output_dim)
        self.b2 = np.zeros(self.output_dim)
        

    # -------------------------------------------------
    # Forward pass components
    # -------------------------------------------------
    def sigmoid(self, a):
        return 1.0 / (1.0 + np.exp(-a))

    def forward(self, X):
        """
        Forward pass. Returns activations dict and output.
        """
        a1 = X @ self.W1 + self.b1
        z1 = self.sigmoid(a1)
        a2 = z1 @ self.W2 + self.b2
        y_hat = self.sigmoid(a2)

        activations = {
            "X": X,
            "a1": a1,
            "z1": z1,
            "a2": a2,
            "y_hat": y_hat
        }
        return activations, y_hat


    # -------------------------------------------------
    # Loss
    # -------------------------------------------------
    def compute_loss(self, y_hat, y):
        """MSE loss."""
        return 0.5 * np.mean((y_hat - y) ** 2)

    # -------------------------------------------------
    # Backward pass (gradient computation)
    # -------------------------------------------------
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid: σ'(a) = σ(a) * (1 - σ(a)) = z * (1 - z)"""
        return z * (1.0 - z)
    
    def backward(self, y, activations):
        """
        Backpropagation. Returns grads dict with W1, b1, W2, b2.
        """
        X = activations["X"]
        z1 = activations["z1"]
        y_hat = activations["y_hat"]
        N = X.shape[0]

        # Output layer error
        delta2 = (y_hat - y) * self.sigmoid_derivative(y_hat)

        # Gradients for W2 and b2
        dW2 = z1.T @ delta2 / N
        db2 = np.mean(delta2, axis=0)

        # Hidden layer error
        delta1 = (delta2 @ self.W2.T) * self.sigmoid_derivative(z1)

        # Gradients for W1 and b1
        dW1 = X.T @ delta1 / N
        db1 = np.mean(delta1, axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return grads

    # -------------------------------------------------
    # Gradient descent step (EXPLICIT)
    # -------------------------------------------------
    def gradient_step(self, grads):
        """Performs one gradient descent update."""
        self.W1 -= self.lr * grads["W1"]
        self.b1 -= self.lr * grads["b1"]
        self.W2 -= self.lr * grads["W2"]
        self.b2 -= self.lr * grads["b2"]

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    def fit(self, X, y, batch_size=8):
        """
        Train the model using mini-batch gradient descent.
        """
        self.reset_weights()
        self.loss_history = []
        N = X.shape[0]

        for epoch in range(self.epochs):
            perm = np.random.permutation(N)
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            epoch_loss = 0.0

            for start in range(0, N, batch_size):
                end = start + batch_size
                Xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]

                activations, y_hat = self.forward(Xb)
                batch_loss = self.compute_loss(y_hat, yb)
                epoch_loss += batch_loss * len(Xb)
                
                grads = self.backward(yb, activations)
                self.gradient_step(grads)

            self.loss_history.append(epoch_loss / N)
        
        return self.loss_history

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    def predict(self, X):
        _, y_hat = self.forward(X)
        return np.argmax(y_hat, axis=1)
    
    # --------------------------------------------------
    # predefined weights (for checking forward pass)
    # --------------------------------------------------
    def set_predefined_weights(self):
        W1, b1, W2, b2 = self.predefined_spiral_weights()

        self.W1 = W1.copy()
        self.b1 = b1.copy()
        self.W2 = W2.copy()
        self.b2 = b2.copy()

    @staticmethod
    def predefined_spiral_weights():
        b1 = np.array([13.955, -1.079, -1.420, -9.452, -6.745])
        # Note: This is realized as 
        # - input dimensions in first dimension
        # - hidden layer in second dimension
        # in an np.array.
        # Leads to computation order: Input array multiplied by matrix.
        W1 = np.array([
            [ 5.290,  24.748,  9.823, -22.713, 13.622],
            [21.145, -28.266,  0.001,   1.745,  8.618],
        ])

        b2 = np.array([14.265, -0.281, -25.521])
        W2 = np.array([
            [-17.921,   3.912,  23.039],
            [-10.631,  -5.133,  15.726],
            [ 27.089, -17.227, -18.574],
            [ -0.646, -12.115,   7.881],
            [-24.915,  32.302, -11.516],
        ])

        return W1, b1, W2, b2