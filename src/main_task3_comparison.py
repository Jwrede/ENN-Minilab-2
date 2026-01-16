
import numpy as np
from src.mlp_one_hidden import MLPOneHiddenLayer
from src.visualization import plot_hidden_size_vs_accuracy

# Load data
# --------------------------------------------------
data = np.load("data/spiral_dataset.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Experiment settings
# --------------------------------------------------
hidden_sizes = list(range(1, 21))   # 1 to 20 neurons
lr = 0.05
epochs = 5000
batch_size = 8

train_accs = []
test_accs = []

# Capacity Loop: Generate comparison data
# --------------------------------------------------
# TODO: Iterate over the different hidden sizes
# and collect train and test errors at the end of training 
for h in hidden_sizes:
    print(f"\nTraining MLP with hidden_dim = {h}")

# Plot results
# --------------------------------------------------
plot_hidden_size_vs_accuracy(
    hidden_sizes=hidden_sizes,
    train_accuracies=train_accs,
    test_accuracies=test_accs,
    path="results/hidden_size_vs_accuracy_3.pdf"
)