import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Compute numerically stable softmax probabilities row-wise.
    Each row represents the logits of one sample. The maximum
    logit is subtracted before exponentiation to prevent possible overflows.

    Args:
        z (np.ndarray): Logits with shape (batch_size, num_classes).

    Returns:
        np.ndarray: Class probabilities with the same shape as z.
                    Each row sums to approximately 1.
    """
    z_new = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_new)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute the mean categorical cross-entropy loss.

    Args:
        probs: Predicted class probabilities, shape (batch_size, num_classes).
        targets: One-hot encoded labels with the same shape.

    Returns:
        Mean cross-entropy loss across the batch.
    """
    safe_probs = np.clip(probs, 1e-12, 1.0)
    return - np.mean(np.sum(targets * np.log(safe_probs), axis=1))


class SoftmaxRegression:
    def __init__(self, input_dim: int, num_classes: int):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.W = np.zeros((self.input_dim, self.num_classes))
        self.b = np.zeros(self.num_classes)

    def forward(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W + self.b
        return softmax(logits)

    def compute_gradients(self, X: np.ndarray, Y: np.ndarray):        
        pass

    def fit(self):
        pass