import json
from pathlib import Path


def save_experiment(name, config, metrics, history):
    """
    Save experiment configuration, evaluation metrics,
    and training history as JSON files.
    """
    experiment_dir = Path("experiments") / name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "config.json": config,
        "metrics.json": metrics,
        "history.json": history,
    }

    for filename, data in files.items():
        filepath = experiment_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    return experiment_dir