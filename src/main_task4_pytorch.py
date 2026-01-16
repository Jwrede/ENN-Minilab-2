
from src.torch_train_loop import train_model
from src.visualization import plot_mean_learning_curve, plot_accuracy_comparison

# --------------------------------------------------
# Train models
# --------------------------------------------------
# TODO 1: In train_model (whcih is given) you have to implement the
# pytorch model
loss_sgd_1, acc_sgd_1 = train_model(batch_size=1, optimizer_name="sgd", runs=3, epochs=2500, lr=0.05)
loss_sgd_8, acc_sgd_8 = train_model(batch_size=8, optimizer_name="sgd", runs=3, epochs=2500, lr=0.05)

# --------------------------------------------------
# Plot individual learning curves
# --------------------------------------------------
plot_mean_learning_curve(loss_sgd_1, acc_sgd_1, "SGD batch=1", "results/sgd_b1.pdf")
plot_mean_learning_curve(loss_sgd_8, acc_sgd_8, "SGD batch=8", "results/sgd_b8.pdf")

# --------------------------------------------------
# Plot accuracy comparison for different batch sizes
# --------------------------------------------------
# TODO 2: You have to implement the plot_accuracy_comparison function
plot_accuracy_comparison(
    acc_sgd_1,
    acc_sgd_8,
    label_1="SGD batch=1",
    label_2="SGD batch=8",
    path="results/accuracy_batch_comparison.pdf"
)