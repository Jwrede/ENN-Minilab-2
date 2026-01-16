
import numpy as np
from src.mlp_one_hidden import MLPOneHiddenLayer
from src.visualization import plot_training_curve

# Load data
# --------------------------------------------------
data = np.load("data/spiral_dataset.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Initialize MLP (2 -> 5 -> 3)
# --------------------------------------------------
model = MLPOneHiddenLayer(
        hidden_dim=5,
        lr=0.05,
        epochs=5000
)

# Forward pass with predefined weights
# --------------------------------------------------
model.set_predefined_weights()
acc = (model.predict(X_test) == np.argmax(y_test, axis=1)).mean()
print("Accuracy predefined weights: ", acc)

# Reset weights and train normally
# --------------------------------------------------
model.reset_weights()
acc = (model.predict(X_test) == np.argmax(y_test, axis=1)).mean()
print("Accuracy before training: ", acc)
model.fit(X_train, y_train, batch_size=8)
acc = (model.predict(X_test) == np.argmax(y_test, axis=1)).mean()
print("Accuracy after training:  ", acc)

plot_training_curve(model.loss_history, "results/training_curve_2.pdf")
