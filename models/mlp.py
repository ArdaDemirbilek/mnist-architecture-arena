import numpy as np
from models.softmax_regression import softmax


def relu(X: np.ndarray) -> np.ndarray:
    """
    Apply the ReLU activation element-wise, replacing
    negative values with zero.

    Args:
        X: Input array of any shape.

    Returns:
        Array with the same shape as X and all values >= 0.
    """
    return  np.maximum(0, X)


class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        self.rng = np.random.default_rng(42)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.W1 = self.rng.normal(
            loc=0.0,
            scale=np.sqrt(2 / input_dim),
            size=(input_dim, hidden_dim)
            )
        self.W2 = self.rng.normal(
            loc=0.0,
            scale=np.sqrt(2 / hidden_dim),
            size=(hidden_dim, num_classes)
            )
        self.b1 = np.zeros((hidden_dim, ))
        self.b2 = np.zeros((num_classes, ))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute class probabilities by passing the input through
        a ReLU hidden layer and a softmax output layer.

        Args:
            X: Input batch of shape (batch_size, input_dim).

        Returns:
            Class probabilities of shape (batch_size, num_classes).
        """
        z1 = X @ self.W1 + self.b1
        l1 = relu(z1)
        z2 = l1 @ self.W2 + self.b2
        l2 = softmax(z2)
        return l2

    def compute_gradients(self, X: np.ndarray, Y: np.ndarray):
        ...

    def fit(self, X_train, Y_train, epochs=20, batch_size=64, learning_rate=0.01):
        ...