from typing import Dict,  Any, Callable, Optional

import numpy as np
import numpy.typing as npt

from utils import calculate_f1_score
from constants import TYPE_VANILLA_PERPCETRON, TYPE_AVERAGED_PERCEPTRON, RANDOM_SEED

class VanillaPerceptron:
    def __init__(
        self,
        max_iterations: int = 1000,
        learning_rate: float = 1e-2,
        shuffle: bool = True,
        class_weights: dict = None,
        lr_scheduler_func: Optional[Callable[[float, int], float]] = None,
        score_func: Callable[[npt.NDArray, npt.NDArray], float] = calculate_f1_score,
        rng=np.random.default_rng(seed=RANDOM_SEED),
        debug: bool = False,
        debug_at: int = 50,
    ) -> None:
        self.type = TYPE_VANILLA_PERPCETRON
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.shuffle = shuffle
        self.class_weights = class_weights
        self.learning_rate_scheduler = lr_scheduler_func
        self.calculate_score = score_func
        self.rng = rng
        self.debug = debug
        self.debug_at = debug_at

        self.best_epoch = 0

    def fit(
        self,
        X_train: npt.NDArray,
        y_train: npt.NDArray,
        X_val: npt.NDArray = None,
        y_val: npt.NDArray = None,
    ):
        n_epoch = 0

        self.weights: npt.NDArray = self.rng.random(size=(X_train.shape[-1],))
        self.bias: float = 0.0

        learning_rate = self.learning_rate

        best_epoch = 0
        best_val_score = -1
        best_weights = self.weights.copy()
        best_bias = self.bias

        for n_epoch in range(1, self.max_iterations + 1):

            if self.shuffle:
                idxs = np.arange(X_train.shape[0])
                self.rng.shuffle(idxs)

                X_train = X_train[idxs]
                y_train = y_train[idxs]

            for x, y_true in zip(X_train, y_train):

                if y_true * self._activation(x) <= 0:
                    if self.class_weights is None:
                        self.weights = self.weights + y_true * x * self.learning_rate
                    else:
                        self.weights = self.weights + y_true * x * self.class_weights[y_true] * self.learning_rate
                    self.bias = self.bias + y_true

            if X_val is not None and y_val is not None:
                train_score = self.calculate_score(y_train, self.predict(X_train))
                val_score = self.calculate_score(y_val, self.predict(X_val))

                if val_score > best_val_score:
                    best_val_score = val_score

                    # Record the current best wegiths and bias
                    best_epoch = n_epoch
                    best_weights = self.weights
                    best_bias = self.bias

                if self.debug and (n_epoch == self.max_iterations or n_epoch % self.debug_at == 0):
                    print("Epoch #", n_epoch, " Train: ", train_score, " Val: ", val_score)
            else:
                best_epoch = n_epoch
                best_weights = self.weights
                best_bias = self.bias

            # Update learning rate
            if self.learning_rate_scheduler:
                learning_rate = self.learning_rate_scheduler(learning_rate, n_epoch)

        # Set the best weights and bias found
        self.best_epoch = best_epoch
        self.weights = best_weights
        self.bias = best_bias

    def _activation(self, x: npt.NDArray):
        return np.dot(self.weights, x) + self.bias

    def predict(self, X: npt.NDArray):
        predictions = list()
        for x in X:
            pred = np.sign(self._activation(x))
            predictions.append(pred)
        return np.array(predictions)

    def export(
        self,
    ):
        return {
            "type": self.type,
            "max_iterations": self.best_epoch,
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "best_epoch": self.best_epoch,
        }

    def load(self, model_data: Dict[str, Any]):
        self.type = model_data["type"]
        self.max_iterations = (model_data["max_iterations"],)
        self.weights = np.array(model_data["weights"])
        self.bias = model_data["bias"]
        self.best_epoch = model_data["best_epoch"]


class AveragedPerceptron:
    def __init__(
        self,
        max_iterations: int = 1000,
        learning_rate: float = 1e-2,
        shuffle: bool = True,
        class_weights: dict = None,
        lr_scheduler_func: Optional[Callable[[float, int], float]] = None,
        score_func: Callable[[npt.NDArray, npt.NDArray], float] = calculate_f1_score,
        rng=np.random.default_rng(seed=RANDOM_SEED),
        debug: bool = False,
        debug_at: int = 50,
    ) -> None:
        self.type = TYPE_AVERAGED_PERCEPTRON
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.shuffle = shuffle
        self.class_weights = class_weights
        self.learning_rate_scheduler = lr_scheduler_func
        self.calculate_score = score_func
        self.rng = rng
        self.debug = debug
        self.debug_at = debug_at

        self.best_epoch = 0

    def fit(
        self,
        X_train: npt.NDArray,
        y_train: npt.NDArray,
        X_val: npt.NDArray = None,
        y_val: npt.NDArray = None,
    ):
        n_epoch = 0

        self.weights = None
        self.bias = None
        current_weights: npt.NDArray = self.rng.random(size=(X_train.shape[-1],))
        current_bias: float = 0.0
        self.cache = {"weights": current_weights, "bias": current_bias}

        c = 1

        best_val_score = -1
        best_epoch = 0
        best_weights = current_weights
        best_bias = current_bias
        best_cache = self.cache

        for n_epoch in range(1, self.max_iterations + 1):

            if self.shuffle:
                idxs = np.arange(X_train.shape[0])
                self.rng.shuffle(idxs)

                X_train = X_train[idxs]
                y_train = y_train[idxs]

            for x, y_true in zip(X_train, y_train):

                a = np.dot(current_weights, x) + current_bias
                if y_true * a <= 0:
                    if self.class_weights is None:
                        current_weights = current_weights + y_true * x * self.learning_rate
                        self.cache["weights"] = self.cache["weights"] + y_true * c * x * self.learning_rate
                    else:
                        current_weights = current_weights + y_true * x * self.learning_rate * self.class_weights[y_true]
                        self.cache["weights"] = (
                            self.cache["weights"] + y_true * c * x * self.learning_rate * self.class_weights[y_true]
                        )

                    current_bias = current_bias + y_true
                    self.cache["bias"] = self.cache["bias"] + y_true * c

                c += 1

            self.weights = current_weights - (1 / c) * self.cache["weights"]
            self.bias = current_bias - (1 / c) * self.cache["bias"]

            if X_val is not None and y_val is not None:
                train_score = self.calculate_score(y_train, self.predict(X_train))
                val_score = self.calculate_score(y_val, self.predict(X_val))

                if val_score > best_val_score:
                    best_val_score = val_score

                    best_epoch = n_epoch
                    best_weights = self.weights
                    best_bias = self.bias
                    best_cache = self.cache

                if self.debug and (n_epoch == self.max_iterations or n_epoch % self.debug_at == 0):
                    print("Epoch #", n_epoch, " Train: ", train_score, " Val: ", val_score)
            else:
                best_epoch = n_epoch
                best_weights = self.weights
                best_bias = self.bias
                best_cache = self.cache

            # Update learning rate
            if self.learning_rate_scheduler:
                learning_rate = self.learning_rate_scheduler(learning_rate, n_epoch)

        # Set best epochs, weight, bias and cache
        self.best_epoch = best_epoch
        self.weights = best_weights
        self.bias = best_bias
        self.cache = best_cache

    def _activation(self, x: npt.NDArray):
        return np.dot(self.weights, x) + self.bias

    def predict(self, X: npt.NDArray):
        predictions = list()
        for x in X:
            pred = np.sign(self._activation(x))
            predictions.append(pred)
        return np.array(predictions)

    def export(
        self,
    ):
        return {
            "type": self.type,
            "max_iterations": self.max_iterations,
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "best_epoch": self.best_epoch,
        }

    def load(self, model_data: Dict[str, Any]):
        self.type = model_data["type"]
        self.max_iterations = (model_data["max_iterations"],)
        self.weights = np.array(model_data["weights"])
        self.bias = model_data["bias"]
        self.best_epoch = model_data["best_epoch"]
