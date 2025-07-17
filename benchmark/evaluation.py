import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def gen_confusion_matrix(output_dir_str: str, run_tool_output_dir_str: str):
    """
    Generates a confusion matrix from the data in 'run-tool-output/confusion-matrx.json'
    and saves it as a PNG image in the specified output directory.
    """
    confusion_matrix_path = Path(run_tool_output_dir_str) / "confusion-matrx.json"
    output_dir = Path(output_dir_str)
    output_dir.mkdir(exist_ok=True)

    with open(confusion_matrix_path, "r") as f:
        data = json.load(f)

    y_true = data["y_true"]
    y_pred = data["y_pred"]

    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    display_labels = ["Buggy", "Correct"]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues)

    plt.title("Confusion Matrix")
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()
