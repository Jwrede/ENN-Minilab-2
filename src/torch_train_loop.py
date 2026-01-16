
import torch
import numpy as np
import torch.nn as nn
from src.torch_mlp import TorchMLP
from torch.utils.data import TensorDataset, DataLoader

# --------------------------------------------------
# Data loading
# --------------------------------------------------
def load_spiral_npz(batch_size=8, shuffle=True):
    """
    Loads the spiral dataset from the npz file and wraps it into PyTorch DataLoaders.

    The DataLoader:
      - handles batching automatically
      - optionally shuffles the data each epoch
      - returns mini-batches (xb, yb).

    You don't have to change or adapt this.
    """

    # Load NumPy data
    import os
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'spiral_dataset.npz')
    data = np.load(data_path)

    # Convert inputs to torch tensors
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    X_test  = torch.tensor(data["X_test"], dtype=torch.float32)

    # Convert one-hot labels to class indices (0,1,2)
    y_train = torch.tensor(np.argmax(data["y_train"], axis=1), dtype=torch.long)
    y_test  = torch.tensor(np.argmax(data["y_test"], axis=1), dtype=torch.long)

    # Wrap tensors into PyTorch datasets
    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test, y_test)

    # DataLoader creates iterable mini-batches
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_loader  = DataLoader(test_ds, batch_size=len(test_ds))

    return train_loader, test_loader

# --------------------------------------------------
# Training loop
# --------------------------------------------------
def train_model(batch_size=8, optimizer_name="sgd", runs=1, epochs=2500, lr=0.05):
    """
    Trains an MLP multiple times (runs) and records loss, accuray for each run.

    Parameters:
      batch_size     : mini-batch size used by the DataLoader
      optimizer_name : "sgd" or "adam", you can add further optimizer
      runs           : number of independent training runs
      epochs         : training epochs per run (you will need more for convergence)
      lr             : learning rate

    Returns:
      all_losses      : list of loss curves (one list per run)
      all_accuracies  : list of accuracy curves (one list per run)
    """

    all_losses = []
    all_accuracies = []

    # --------------------------------------------------
    # Multiple independent runs
    # --------------------------------------------------
    for _ in range(runs):

        # Load data (DataLoader handles batching)
        train_loader, _ = load_spiral_npz(batch_size=batch_size)

        # Initialize model
        # TODO: You might choose a different size here
        model = TorchMLP(hidden_dim=5)

        # Mean Squared Error
        criterion = nn.MSELoss()

        # Choose optimizer
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        run_loss = []
        run_acc = []

        # --------------------------------------------------
        # Training loop
        # --------------------------------------------------
        for _ in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            # --------------------------------------------------
            # Mini-batch loop (handled by DataLoader)
            # --------------------------------------------------
            for xb, yb in train_loader:

                # Convert class labels to one-hot (to match MSE loss)
                y_onehot = torch.zeros(len(yb), 3)
                y_onehot[range(len(yb)), yb] = 1.0

                # Reset accumulated gradients
                optimizer.zero_grad()

                # Forward pass
                out = model(xb)

                # Compute loss
                loss = criterion(out, y_onehot)

                # Backward pass (automatic backpropagation)
                loss.backward()

                # Gradient descent step
                optimizer.step()

                # Accumulate loss
                epoch_loss += loss.item()

                # Compute accuracy
                preds = torch.argmax(out, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

            # Store epoch statistics
            run_loss.append(epoch_loss)
            run_acc.append(correct / total)

        # Store results of this run
        all_losses.append(run_loss)
        all_accuracies.append(run_acc)

    # Return: two nested lists = each of the variables contains for each run a list of losses / accuracies
    return all_losses, all_accuracies
