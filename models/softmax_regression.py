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
