"""
Test file for Task 1 — Simple Neural Network (Backpropagation)
--------------------------------------------------------------
This file validates:

  1. Loading of the spiral dataset (train / test split)
  2. Training of a simple neural network without hidden layer
  3. Test accuracy above a minimum threshold
  4. Creation of visualization outputs:
     - decision regions
     - training curve

Run manually via:
    python -m pytest -s tests/test_1_simple_nn.py
"""

import os
import numpy as np

from src.simple_nn import SimpleNeuralNetwork
from src.visualization import plot_training_curve, plot_decision_regions


def test_simple_nn_training_and_visualization():
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data = np.load("data/spiral_dataset.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # ------------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------------
    model = SimpleNeuralNetwork(epochs=200)
    history = model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluate accuracy
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)

    acc = (y_pred == y_true).mean()
    print(f"\n✅ Test accuracy on spiral dataset: {acc:.3f}")

    assert acc > 0.5, f"Accuracy too low: {acc:.3f}"

    # ------------------------------------------------------------------
    # Visualization checks
    # ------------------------------------------------------------------
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Plot training curve
    path_tc = plot_training_curve(
        model.loss_history,
        path=os.path.join(output_dir, "task1_training_curve.pdf")
    )

    # ---- checks ----
    assert os.path.exists(path_tc), "Training curve file was not created"
    assert os.path.getsize(path_tc) > 0, "Training curve file is empty"

    print(f"✅ Training curve successfully created at: {path_tc}")

    path_dr = plot_decision_regions(model, X_train, y_train, path=os.path.join(output_dir, "task1_decision_regions.pdf"))

    assert os.path.exists(path_dr), "Decision region plot was not created"
    assert os.path.getsize(path_dr) > 0, "Decision region plot is empty"

    print(f"✅ Decision region visualization created: {path_dr}")


#    curve_pdf = plot_training_curve(
 #       history,
#        "Task_1_training_curve.pdf"
  #  )

    # Check files exist
    #assert os.path.exists(decision_pdf), "Decision region plot not created" 
   # assert os.path.exists(curve_pdf), "Training curve plot not created"

    #print("✅ Visualization files created successfully:")
    #print(f" - {decision_pdf}")
    #print(f" - {curve_pdf}")

