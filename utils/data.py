from pathlib import Path
from urllib.request import urlretrieve

import numpy as np


MNIST_URL = (
    "https://storage.googleapis.com/"
    "tensorflow/tf-keras-datasets/mnist.npz"
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_PATH = DATA_DIR / "mnist.npz"


def load_mnist(validation_size=5000, seed=42):
    """
    Load and preprocess MNIST.

    Returns:
        X_train, Y_train: Training features and one-hot labels.
        X_val, Y_val: Validation features and one-hot labels.
        X_test, Y_test: Test features and one-hot labels.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        print("Downloading MNIST dataset...")
        urlretrieve(MNIST_URL, DATA_PATH)

    with np.load(DATA_PATH) as data:
        X_train = data["x_train"]
        y_train = data["y_train"]
        X_test = data["x_test"]
        y_test = data["y_test"]

    # Flatten images and normalize pixels to [0, 1].
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float64) / 255.0

    # Shuffle before creating the validation split.
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X_train))

    val_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    X_val = X_train[val_indices]
    X_train = X_train[train_indices]

    y_val = y_train[val_indices]
    y_train = y_train[train_indices]

    # One-hot encode labels.
    Y_train = np.eye(10)[y_train]
    Y_val = np.eye(10)[y_val]
    Y_test = np.eye(10)[y_test]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test