"""
Test file for Task 4 — PyTorch MLP Experiments
-----------------------------------------------------------
This file validates:

  1) Learning-curve generation:
     A provided function trains a PyTorch MLP with a given
     batch size and optimizer (possibly multiple runs) and
     returns learning curves of the correct structure.

  2) Batch-size comparison visualization:
     Learning curves for at least two different batch sizes
     are plotted together and saved as a file.

Run manually via:
    python -m pytest -s tests/test_4_pytorch_mlp.py
"""

import os
import numpy as np

from src.torch_train_loop import train_model
from src.visualization import plot_accuracy_comparison, plot_mean_learning_curve

# ---------------------------------------------------------------------
# 4.1 Test: Training model and learning curve generation (single configuration)
# ---------------------------------------------------------------------
def test_learning_curve_generation():
    # Train model is already defined
    # You have to work inside of the model files and setup the model.
    losses, accs = train_model(
        batch_size=8,
        optimizer_name="sgd",
        runs=2,
        epochs=20,      # keep test fast
        lr=0.05
    )

    print(f"📈 Received {len(losses)} runs of learning curves")

    # ---- structural checks ----
    assert isinstance(losses, list), "Losses must be returned as a list"
    assert isinstance(accs, list), "Accuracies must be returned as a list"

    assert len(losses) == 2, "Expected one loss curve per run"
    assert len(accs) == 2, "Expected one accuracy curve per run"

    assert len(losses[0]) == 20, "Loss curve length must match epochs"
    assert len(accs[0]) == 20, "Accuracy curve length must match epochs"

    for i in range(0, len(losses[0])-1):
        assert losses[0][1] > losses[0][1+1] , "Loss decreases"

    # Plotting function is already given - just generates results as a simple check
    out_path = "results/task4_sgd_b1_learning_curve.pdf"

    plot_mean_learning_curve(losses, accs, "SGD_batch8", out_path)

    assert os.path.exists(out_path), (
        "Learning curveplot was not created."
    )
    assert os.path.getsize(out_path) > 0, (
        "Learning curve plot file is empty."
    )

    print("✅ Training model and learning-curve generation test passed.")

# ---------------------------------------------------------------------
# 4.2 Test: Batch-size comparison visualization
# ---------------------------------------------------------------------
def test_batch_size_comparison_plot():
    os.makedirs("results", exist_ok=True)

    # Train two configurations
    losses_b1, accs_b1 = train_model(
        batch_size=1,
        optimizer_name="sgd",
        runs=2,
        epochs=100,
        lr=0.05
    )

    losses_b8, accs_b8 = train_model(
        batch_size=8,
        optimizer_name="sgd",
        runs=2,
        epochs=100,
        lr=0.05
    )

    out_path = "results/task4_batch_size_comparison.pdf"

    # Plot comparison: 
    # You have to setup this function that visualizes the comparison for two 
    # different training runs.
    plot_accuracy_comparison(accs_b1, accs_b8,
        label_1="SGD batch=1", label_2="SGD batch=8",
        path=out_path
    )

    assert os.path.exists(out_path), (
        "Batch-size comparison plot was not created."
    )
    assert os.path.getsize(out_path) > 0, (
        "Batch-size comparison plot file is empty."
    )

    print(f"✅ Batch-size comparison plot created: {out_path}")
