import re

import json
from typing import List, Tuple, Any

import numpy as np
import numpy.typing as npt

from constants import RANDOM_SEED


# Load Data
# Load Data
def load_data(input_file_path: str, type: str = "TRAIN") -> npt.NDArray:
    input_data = list()
    regex = r"(\w*) (\w*) (\w*) (.*)\n?"
    if type == "DEV":
        regex = r"(\w*) (.*)\n?"
    elif type == "KEY":
        regex = r"(\w*) (\w*) (\w*)\n?"
    input_regex = re.compile(regex)
    with open(input_file_path, mode="r") as input_file:
        for line in input_file:
            input_data.append(re.match(input_regex, line).groups())
    return np.array(input_data)


# Store Data
def store_data(date_file_path: str, data: npt.NDArray) -> None:
    with open(date_file_path, mode="w") as data_file:
        for row in data:
            data_file.write(f"{row[0]} {row[1]} {row[2]} {row[3]}\n")


# Store Model
def store_model(model_file_path: str, model_data: Any) -> None:
    with open(model_file_path, mode="w") as model_file:
        json.dump(model_data, model_file, ensure_ascii=False)


# Load Model
def load_model(model_file_path: str) -> npt.NDArray:
    with open(model_file_path, mode="r") as model_file:
        model_data = json.load(model_file)

    return (model_data["tf_idf_model"], model_data["sentiment_classifier"], model_data["truthfulness_classifier"])


# Store Predictions
def store_predictions(output_file_path: str, predictions: List[Tuple[str, str, str]]) -> None:
    with open(output_file_path, mode="w") as output_file:
        for prediction in predictions:
            output_file.write(f"{prediction[0]} {prediction[1]} {prediction[2]}\n")


def calculate_accuracy_score(y_true: npt.NDArray, y_pred: npt.NDArray):
    return (y_true == y_pred).sum() / y_true.shape[0]


def calculate_f1_score(y_true: npt.NDArray, y_pred: npt.NDArray, average: str = "macro"):
    def calculate_f1(y_true, y_pred, label):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_pred != label) & (y_true == label))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def macro_f1(y_true, y_pred):
        return np.mean([calculate_f1(y_true, y_pred, label) for label in np.unique(y_true)])

    def micro_f1(y_true, y_pred):
        return {label: calculate_f1(y_true, y_pred, label) for label in np.unique(y_true)}

    if average == "macro":
        return macro_f1(y_true, y_pred)
    elif average == "micro":
        return micro_f1(y_true, y_pred)
    else:
        return {"micro": micro_f1(y_true, y_pred), "macro": macro_f1(y_true, y_pred)}


# Calculate Scores
def calculate_scores(y_true, y_pred, title: str):
    from sklearn.metrics import classification_report

    print(f"------------------------ {title} ------------------------")
    print(classification_report(y_true, y_pred))
    print("---------------------------------------------------------")


def train_test_split(
    X: npt.NDArray,
    y_sentiment: npt.NDArray,
    y_truthfulness: npt.NDArray,
    test_size: float = 0.2,
    rng=np.random.default_rng(seed=RANDOM_SEED),
):
    n_max = X.shape[0]
    sample = int((1 - test_size) * n_max)

    # Shuffle the data
    all_idx = np.arange(n_max)
    rng.shuffle(all_idx)

    train_idx, test_idx = all_idx[:sample], all_idx[sample:]

    X_train, X_test, y_train_sentiment, y_test_sentiment, y_train_truthfulness, y_test_truthfulness = (
        X[train_idx],
        X[test_idx],
        y_sentiment[train_idx],
        y_sentiment[test_idx],
        y_truthfulness[train_idx],
        y_truthfulness[test_idx],
    )

    return X_train, X_test, y_train_sentiment, y_test_sentiment, y_train_truthfulness, y_test_truthfulness



# Learning Rate Scheduler
def learning_rate_scheduler(learning_rate: float, epoch: int, decay: float = 1e-2):
    return learning_rate * 1 / (1 + decay * epoch)
