
import numpy as np
from src.simple_nn import SimpleNeuralNetwork
from src.visualization import plot_training_curve

# Load data
# --------------------------------------------------
data = np.load("data/spiral_dataset.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Call of the simple model: Init and fit
# --------------------------------------------------
model = SimpleNeuralNetwork()
model.fit(X_train, y_train)

# Calculating accuracy by hand
acc = (model.predict(X_test) == np.argmax(y_test, axis=1)).mean()
print("Accuracy:", acc)

# Testing the plotting function
plot_training_curve(model.loss_history, "results/training_curve.pdf")
