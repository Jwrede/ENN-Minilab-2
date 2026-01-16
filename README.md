# Introduction to Neural Networks, Mini-Lab 2 – Classification, Backpropagation, and MLPs

This repository contains the materials for the second minilab of the course  
*Introduction to Neural Networks (WS 2025)* at the University of Münster.

The goal of this mini-lab is to move from **simple neural classifiers** to  
**multilayer perceptrons (MLPs)** and to develop a **practical understanding of backpropagation**,  
model capacity, and generalization on a **non-linearly separable dataset**.

---

## Learning Goals

This exercise should enable you to

- Work with a **synthetic multi-class classification dataset** (three-class spiral)
- Implement a **simple neural classifier** without hidden layers
- Derive and implement **backpropagation** step by step
- Extend the model to a **Multilayer Perceptron (MLP)** with one hidden layer
- Analyze the **effect of model capacity** (hidden layer size)
- Compare **training behavior and generalization**
- Implement and visualize **decision regions and learning curves**
- Transfer the implementation to **PyTorch** and compare optimizers and batch sizes

The detailed task descriptions can be found in  
📄 `enn_minilab_2_classification.pdf`

---

## Task Overview

| Task | Topic                           | Core Idea |
| ---- | ------------------------------- | --------- |
| **1** | Simple Neural Network (Baseline) | Implement a single-layer neural classifier; visualize learning and decision regions |
| **2** | Backpropagation & MLP           | Extend the model to one hidden layer; derive and implement backpropagation |
| **3** | Model Capacity & Generalization | Vary hidden layer size; analyze accuracy, overfitting, and model selection |
| **4** | PyTorch Implementation          | Reimplement the MLP in PyTorch; compare optimizers and batch sizes |

---

## Dataset

All tasks are based on a **three-class spiral dataset**, a standard benchmark for  
non-linear classification.

- Input: 2-dimensional points  
- Output: 3 classes (one-hot encoded)  
- Property: **not linearly separable**

The dataset is provided as a single NumPy archive:

```
data/spiral_dataset.npz
```

containing

- `X_train`, `y_train`
- `X_test`, `y_test`

---

## Folder Structure

| Folder / File                    | Description |
| -------------------------------- | ----------- |
| `data/`                          | Spiral dataset (`.npz`, train/test split) |
| `notebooks/`                     | Optional exploratory notebooks |
| `src/`                           | Model implementations (Simple NN, MLP, PyTorch MLP, visualization) |
| `tests/`                         | Test scripts verifying correctness (recommended starting point) |
| `results/`                       | Output directory for plots (learning curves, decision regions) |
| `enn_minilab_2_classification.pdf` | PDF version of the exercise sheet |
| `README.md`                      | This project overview file |

---

## Running Tests

All implementations can be validated via:

```bash
pytest -m pytest -s tests/...
```

You are encouraged to inspect the tests early to understand what functionality is expected (tests for pytorch will be provided after the exercise sessions).

---

## Notes

Interfaces are already quite broadly provided - you can extend those but should try to stick with the basic interace (and are required to provide the functionality that is called by the tests).
