import numpy as np

from models.softmax_regression import softmax


def test_softmax_probability_normalization():
    logits = np.array([
        [2.0, 1.0, 0.0],
        [0.5, 1.5, 3.0]
    ])
    probabilities = softmax(logits)
    row_sums = np.sum(probabilities, axis=1)

    assert np.allclose(row_sums, 1.0)


def test_softmax_shape_prevention():
    logits = np.array([
        [1000.1, 1000.2, 1000.4],
        [550.9, 999.9, 1000.0],
        [1500.0, 0.0, 250.5]
    ])

    probabilities = softmax(logits)

    assert np.shape(logits) == np.shape(probabilities)


def test_softmax_numerical_stability():
    logits = np.array([
        [1000.1, 1000.2, 1000.4],
        [550.9, 999.9, 1000.0],
        [1500.0, 0.0, 250.5]
    ])
    probabilities = softmax(logits)

    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_softmax_shift_variance():
    logits = np.array([
        [3.5, 1000.2, 4.4],
        [0.5, 7.8, 6.3],
        [3.0, 0.0, 9.9]
    ])
    logits_added = logits + 1.0

    probabilities_normal = softmax(logits)
    probabilities_added = softmax(logits_added)

    assert np.allclose(probabilities_normal, probabilities_added)
