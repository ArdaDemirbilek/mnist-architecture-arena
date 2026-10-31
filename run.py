import numpy as np

from models.softmax_regression import SoftmaxRegression, cross_entropy
from utils.data import load_mnist
from utils.plots import plot_training_history
from utils.experiments import save_experiment


def main():
    # 1. Experiment configuration
    config = {
        "model": "SoftmaxRegression",
        "input_dim": 784,
        "num_classes": 10,
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 0.1,
        "seed": 42,
        "train_samples": 55000,
        "validation_samples": 5000,
    }

    # 2. Load MNIST dataset
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_mnist()

    print(f"Training:   {X_train.shape}, {Y_train.shape}")
    print(f"Validation: {X_val.shape}, {Y_val.shape}")
    print(f"Test:       {X_test.shape}, {Y_test.shape}")

    # 3. Initialize model
    model = SoftmaxRegression(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
    )

    # 4. Calculate initial training loss
    initial_probs = model.forward(X_train)
    initial_loss = float(cross_entropy(initial_probs, Y_train))

    print(f"\nInitial Loss: {initial_loss:.4f}")

    # 5. Train model
    history = model.fit(
        X_train,
        Y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
    )

    # 6. Evaluate on validation set
    val_probs = model.forward(X_val)

    val_predictions = np.argmax(val_probs, axis=1)
    val_labels = np.argmax(Y_val, axis=1)

    val_accuracy = float(np.mean(val_predictions == val_labels))
    val_loss = float(cross_entropy(val_probs, Y_val))

    # 7. Collect evaluation metrics
    metrics = {
        "initial_training_loss": initial_loss,
        "final_training_loss": float(history["train_loss"][-1]),
        "validation_accuracy": val_accuracy,
        "validation_loss": val_loss,
    }

    print(f"Final Training Loss: {metrics['final_training_loss']:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2%}")
    print(f"Validation Loss: {val_loss:.4f}")

    # 8. Save experiment
    experiment_dir = save_experiment(
        name="softmax_baseline",
        config=config,
        metrics=metrics,
        history={
            "train_loss": [
                float(loss) for loss in history["train_loss"]
            ]
        },
    )

    # 9. Plot and save training loss
    plot_training_history(
        history,
        save_path=experiment_dir / "training_loss.png",
    )

    print(f"\nExperiment saved to: {experiment_dir}")


if __name__ == "__main__":
    main()