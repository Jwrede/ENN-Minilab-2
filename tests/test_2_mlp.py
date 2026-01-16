"""
Test file for Task 2 — MLP (One Hidden Layer)
-----------------------------------------------------------
This file validates:
  1) Backpropagation step reduces the loss:
     A single gradient step (backward(...) + gradient_step(...)) must
     reduce the MSE loss on a small batch.
  2) Classification accuracy:
     After full training, the MLP with hidden_dim=5 must achieve
     at least 90% accuracy on the spiral test set (via predict(...)).

Run manually via:
    python -m pytest -s tests/test_2_mlp.py
"""

import os
import numpy as np

from src.mlp_one_hidden import MLPOneHiddenLayer



def load_spiral_npz(path="data/spiral_dataset.npz"):
    data = np.load(path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    return X_train, y_train, X_test, y_test


def mse_loss(y_hat, y):
    return 0.5 * np.mean((y_hat - y) ** 2)


# ---------------------------------------------------------------------
# 2.1 Test: Single backprop step reduces loss
# ---------------------------------------------------------------------
def test_backprop_step_reduces_loss():
    X_train, y_train, _, _ = load_spiral_npz()

    # Small batch to make the test fast + stable
    Xb = X_train[:64]
    yb = y_train[:64]

    model = MLPOneHiddenLayer(hidden_dim=5, lr=0.1, epochs=1)

    # Make sure weights exist (either init in __init__ or call reset here)
    model.reset_weights()

    # Forward -> loss before
    activations, y_hat_before = model.forward(Xb)
    loss_before = mse_loss(y_hat_before, yb)

    # One gradient step
    grads = model.backward(yb, activations)
    model.gradient_step(grads)

    # Forward -> loss after
    _, y_hat_after = model.forward(Xb)
    loss_after = mse_loss(y_hat_after, yb)

    print(f"📉 Loss before step: {loss_before:.6f}")
    print(f"📉 Loss after  step: {loss_after:.6f}")

    assert loss_after < loss_before, (
        "Backprop test failed: a single gradient step did not reduce the loss. "
        "Check backward(...), gradient_step(...), and learning rate."
    )

    print("✅ Backprop step reduces the loss.")


# ---------------------------------------------------------------------
# 2.2 Test: Final model accuracy >= 90% on test set
# ---------------------------------------------------------------------
def test_mlp_accuracy():
    X_train, y_train, X_test, y_test = load_spiral_npz()

    model = MLPOneHiddenLayer(hidden_dim=5, lr=0.1, epochs=5000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = (y_pred == np.argmax(y_test, axis=1)).mean()

    print(f"✅ MLP test accuracy: {acc:.3f}")

    assert acc >= 0.90, (
        f"Accuracy too low: {acc:.3f} < 0.90. "
        "Check training loop (mini-batches/shuffling), gradients, and hyperparameters."
    )

    print("✅ Accuracy requirement satisfied (>= 90%).")