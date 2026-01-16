import matplotlib.pyplot as plt
import numpy as np

# Exercise 1 - plotting the training curve
def plot_training_curve(loss_history, path):
    """
    Plot training loss over epochs and save to file.

    Parameters
    ----------
    loss_history : list or np.ndarray
        Loss values collected during training.
    path : str
        Directory and file name for the output PDF file.

    Returns
    -------
    str
        Path to the saved figure.
    """
    plt.figure(figsize=(8, 5))
    
    epochs = range(1, len(loss_history) + 1)
    plt.plot(epochs, loss_history, 'b-', linewidth=2, label='Training Loss')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path

# Exercise 1 - Visualization of the data set and decision regions
def plot_decision_regions(model, X, y, path):
    """
    Plot decision regions and training data points

    Parameters
    ----------
    model : 
        fitted model, can be used to predict any point in the plane
    X:
        training inputs
    y:
        training targets (can be one-hot or class indices)
    path : str
        Directory and file name for the output PDF file.

    Returns
    -------
    str
        Path to the saved figure.
    """
    plt.figure(figsize=(8, 8))

    margin = 0.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    
    resolution = 200
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    cmap_bg = plt.cm.RdYlBu
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_bg)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)
    
    if len(y.shape) > 1 and y.shape[1] > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y
    
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_labels, cmap=cmap_bg, 
                          edgecolor='k', s=40, linewidth=0.5)
    
    plt.xlabel("$x_1$", fontsize=12)
    plt.ylabel("$x_2$", fontsize=12)
    plt.title("Decision Regions", fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Class')
    
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path

# Exercise 3 - Visualize accuracies depending on the hidden layer size
# at the end of training
def plot_hidden_size_vs_accuracy(
    hidden_sizes,
    train_accuracies,
    test_accuracies,
    path
):
    """
    Plot training and test accuracy as a function of hidden layer size.

    Parameters
    ----------
    hidden_sizes : list[int]
        Number of neurons in the hidden layer.
    train_accuracies : list[float]
        Training accuracies for each hidden size.
    test_accuracies : list[float]
        Test accuracies for each hidden size.
    path : str
        Directory and file name of the output PDF file.

    Returns
    -------
    str
        Path to the saved PDF file.
    """
    plt.figure(figsize=(8, 5))
    
    plt.plot(hidden_sizes, train_accuracies, 'o-', label='Training', color='tab:blue')
    plt.plot(hidden_sizes, test_accuracies, 's--', label='Test', color='tab:orange')
    
    plt.xlabel('Hidden Layer Size', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Capacity vs Accuracy', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path

def plot_mean_learning_curve(losses, accuracies, label, path):
    """
    Plot mean ± std of loss and accuracy over epochs.

    losses:       list or array, shape (n_runs, epochs)
    accuracies:   list or array, shape (n_runs, epochs)
    label:        str (e.g. 'SGD, batch_size=8')
    path:         output PDF path
    """
    losses = np.array(losses)
    accuracies = np.array(accuracies)

    loss_mean = losses.mean(axis=0)
    loss_std = losses.std(axis=0)

    acc_mean = accuracies.mean(axis=0)
    acc_std = accuracies.std(axis=0)

    epochs = np.arange(len(loss_mean))

    fig, ax1 = plt.subplots()

    # ---- loss (left axis) ----
    ax1.plot(epochs, loss_mean, label=f"{label} – loss", color="tab:blue")
    ax1.fill_between(
        epochs,
        loss_mean - loss_std,
        loss_mean + loss_std,
        alpha=0.3,
        color="tab:blue"
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    # ---- accuracy (right axis) ----
    ax2 = ax1.twinx()
    ax2.plot(epochs, acc_mean, label=f"{label} – accuracy", color="tab:orange")
    ax2.fill_between(
        epochs,
        acc_mean - acc_std,
        acc_mean + acc_std,
        alpha=0.3,
        color="tab:orange"
    )
    ax2.set_ylabel("Accuracy")

    # ---- legend ----
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

# Exercise 4: Generate a plot comparing accuracies for different batch sizes.
def plot_accuracy_comparison(acc_list_1, acc_list_2,
                             label_1, label_2,
                             path):
    """
    Compare two accuracy learning curves (mean and std).

    acc_list_1 : list of runs, each run is a list of accuracies over epochs
    acc_list_2 : second list of runs, each run is a list of accuracies over epochs
    """
    acc1 = np.array(acc_list_1)
    acc2 = np.array(acc_list_2)
    
    mean1, std1 = acc1.mean(axis=0), acc1.std(axis=0)
    mean2, std2 = acc2.mean(axis=0), acc2.std(axis=0)
    
    epochs = np.arange(len(mean1))
    
    plt.figure(figsize=(8, 5))
    
    plt.plot(epochs, mean1, label=label_1, color='tab:blue')
    plt.fill_between(epochs, mean1 - std1, mean1 + std1, alpha=0.3, color='tab:blue')
    
    plt.plot(epochs, mean2, label=label_2, color='tab:orange')
    plt.fill_between(epochs, mean2 - std2, mean2 + std2, alpha=0.3, color='tab:orange')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Batch Size Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path
