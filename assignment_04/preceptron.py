from typing import List, Dict, Tuple, Set, Any, Callable

import numpy as np
import numpy.typing as npt

from utils import calculate_f1_score, train_test_split
from constants import TYPE_VANILLA_PERPCETRON, TYPE_AVERAGED_PERCEPTRON

class VanillaPerceptron:
    def __init__(
        self,
        max_iterations: int,
        learning_rate: float = 1e-2,
        tolerance: float = 1e-2,
        val_ratio: float = 0.2,
        shuffle: bool = True,
        class_weights: dict = None,
        debug: bool = False,
        debug_at: int = 50,
        score_func: Callable[[npt.NDArray, npt.NDArray], float] = calculate_f1_score,
    ) -> None:
        self.type= TYPE_VANILLA_PERPCETRON
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.val_ratio = val_ratio
        self.shuffle = shuffle
        self.class_weights = class_weights
        self.debug = debug
        self.debug_at = debug_at
        self.calculate_score = score_func
        self.best_epoch = 0

    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
    ):
        n_epoch = 0

        self.weights: npt.NDArray = np.random.rand(X.shape[-1])
        self.bias: float = 0.0

        X_train, X_val, y_train, y_val = train_test_split(X, y, self.val_ratio)
        best_val_score = -1

        for n_epoch in range(1, self.max_iterations + 1):

            if self.shuffle:
                idxs = np.random.permutation(X.shape[0])
                X = X[idxs]
                y = y[idxs]

            for x, y_true in zip(X_train, y_train):

                a = np.dot(self.weights, x) + self.bias
                if y_true * a <= 0:
                    if self.class_weights is None:
                        self.weights = self.weights + y_true * x * self.learning_rate
                    else:
                        self.weights = self.weights + y_true * x * self.class_weights[y_true] * self.learning_rate
                    self.bias = self.bias + y_true

            if self.val_ratio != 0:
                train_score = self.calculate_score(y_train, self.predict(X_train))
                val_score = self.calculate_score(y_val, self.predict(X_val))

                if val_score > best_val_score:
                    best_val_score = val_score
                    self.best_epoch = n_epoch

                if self.debug and (n_epoch == self.max_iterations or n_epoch % self.debug_at == 0):
                    print("Epoch #", n_epoch, " Train: ", train_score, " Val: ", val_score)

        return self.best_epoch, best_val_score

    def predict(self, X: npt.NDArray):
        predictions = list()
        for x in X:
            pred = np.sign(np.dot(self.weights, x) + self.bias)
            predictions.append(pred)
        return np.array(predictions)

    def export(
        self,
    ):
        return {"type": self.type, "max_iterations": self.best_epoch, "weights": self.weights.tolist(), "bias": float(self.bias)}

    def load(self, model_data: Dict[str, Any]):
        self.type = model_data["type"]
        self.max_iterations = (model_data["max_iterations"],)
        self.weights = np.array(model_data["weights"])
        self.bias = model_data["bias"]



class AveragedPerceptron:
    def __init__(
        self,
        max_iterations: int,
        learning_rate: float = 1e-2,
        tolerance: float = 1e-2,
        val_ratio: float = 0.2,
        shuffle: bool = True,
        class_weights: dict = None,
        debug: bool = False,
        debug_at: int = 50,
        score_func: Callable[[npt.NDArray, npt.NDArray], float] = calculate_f1_score,
    ) -> None:
        self.type = TYPE_AVERAGED_PERCEPTRON
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.val_ratio = val_ratio
        self.shuffle = shuffle
        self.class_weights = class_weights
        self.debug = debug
        self.debug_at = debug_at
        self.calculate_score = score_func
        self.best_epoch = 0

    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
    ):
        n_epoch = 0

        self.weights: npt.NDArray = np.random.rand(X.shape[-1])
        self.bias: float = 0.0

        c = 1
        self.cache = {"weights": np.zeros(shape=(X.shape[-1],)), "bias": 0.0}

        X_train, X_val, y_train, y_val = train_test_split(X, y, self.val_ratio)
        best_val_score = -1

        for n_epoch in range(1, self.max_iterations + 1):

            if self.shuffle:
                idxs = np.random.permutation(X.shape[0])
                X = X[idxs]
                y = y[idxs]

            for x, y_true in zip(X_train, y_train):

                a = np.dot(self.weights, x) + self.bias
                if y_true * a <= 0:
                    if self.class_weights is None:
                        self.weights = self.weights + y_true * x * self.learning_rate
                    else:
                        self.weights = self.weights + y_true * x * self.class_weights[y_true] * self.learning_rate
                    self.bias = self.bias + y_true

                self.cache["weights"] = self.cache["weights"] + y_true * c * x * self.learning_rate
                self.cache["bias"] = self.cache["bias"] + y_true * c

            if self.val_ratio != 0:
                train_score = self.calculate_score(y_train, self.predict(X_train))
                val_score = self.calculate_score(y_val, self.predict(X_val))

                if val_score > best_val_score:
                    best_val_score = val_score
                    self.best_epoch = n_epoch

                if self.debug and (n_epoch == self.max_iterations or n_epoch % self.debug_at == 0):
                    print("Epoch #", n_epoch, " Train: ", train_score, " Val: ", val_score)

        return self.best_epoch, best_val_score

    def predict(self, X: npt.NDArray):
        predictions = list()
        for x in X:
            pred = np.sign(np.dot(self.weights, x) + self.bias)
            predictions.append(pred)
        return np.array(predictions)

    def export(
        self,
    ):
        return {"type": self.type, "max_iterations": self.best_epoch, "weights": self.weights.tolist(), "bias": float(self.bias)}

    def load(self, model_data: Dict[str, Any]):
        self.type = model_data["type"]
        self.max_iterations = (model_data["max_iterations"],)
        self.weights = np.array(model_data["weights"])
        self.bias = model_data["bias"]
