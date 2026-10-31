from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(history, save_path=None):
    """
    Plot training loss over completed epochs.
    """
    losses = history["train_loss"]
    epochs = range(1, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        epochs,
        losses,
        color="#2563eb",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Training Loss"
    )

    ax.set_title("Softmax Regression — Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")

    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    plt.show()