"""
Test file for Task 3 — MLP Model Capacity
----------------------------------------
This test validates:
  Training MLPs with varying hidden-layer sizes
  Collection of training and test accuracies
  Visualization of accuracy vs. model capacity

Run manually via:
    python -m pytest -s tests/test_3_comparison.py
"""

import os
import numpy as np

from src.mlp_one_hidden import MLPOneHiddenLayer
from src.visualization import plot_hidden_size_vs_accuracy


def load_spiral_npz(path="data/spiral_dataset.npz"):
    data = np.load(path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    return X_train, y_train, X_test, y_test

def test_mlp_model_capacity_visualization():
    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    X_train, y_train, X_test, y_test = load_spiral_npz()

    hidden_sizes = [1, 2, 4, 5, 8, 16]
    train_accs = []
    test_accs = []

    # --------------------------------------------------
    # Train models of different capacity
    # --------------------------------------------------
    for h in hidden_sizes:
        model = MLPOneHiddenLayer(
            hidden_dim=h,
            lr=0.1,
            epochs=1500
        )

        model.fit(X_train, y_train, batch_size=1)

        train_acc = (
            model.predict(X_train) == np.argmax(y_train, axis=1)
        ).mean()
        test_acc = (
            model.predict(X_test) == np.argmax(y_test, axis=1)
        ).mean()

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print("Hidden size: ", h, " train acc.: ", train_acc, "; test acc.: ", test_acc)

    # --------------------------------------------------
    # Create visualization
    # --------------------------------------------------
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    plot_path = plot_hidden_size_vs_accuracy(
        hidden_sizes=hidden_sizes,
        train_accuracies=train_accs,
        test_accuracies=test_accs,
        path=os.path.join(output_dir, "task3_capacity_comparison.pdf")
    )

    # --------------------------------------------------
    # Checks
    # --------------------------------------------------
    assert os.path.exists(plot_path), "Capacity plot was not created"
    assert os.path.getsize(plot_path) > 0, "Capacity plot file is empty"

    print(
        f"✅ Capacity plot created successfully for hidden sizes {hidden_sizes}\n"
        f"   File: {plot_path}"
    )