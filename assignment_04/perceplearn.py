import sys
from functools import partial
import numpy as np

from constants import *
from data_cleaning import *
from perceptron import *
from tfidf import *
from utils import *


rng = np.random.default_rng(seed=RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def main(input_file_path: str):

    data = load_data(input_file_path, type="TRAIN")

    # A dictionary containing the columns and a list of functions to perform on it in order
    data_cleaning_pipeline = {
        TRAIN_DATA_COL: [
            to_lower,
            remove_html_encodings,
            remove_html_tags,
            remove_url,
            fix_contractions,
            remove_non_alpha_characters,
            remove_extra_spaces,
        ]
    }

    cleaned_data = data.copy()

    # Process all the cleaning instructions
    for col, pipeline in data_cleaning_pipeline.items():
        # Get the column to perform cleaning on
        temp_data = cleaned_data[:, col].copy()

        # Perform all the cleaning functions sequencially
        for func in pipeline:
            print(f"Starting: {func.__name__}")
            temp_data = func(temp_data)
            print(f"Ended: {func.__name__}")

        # Replace the old column with cleaned one.
        cleaned_data[:, col] = temp_data.copy()

    train_tokenized = tokenize(cleaned_data[:, TRAIN_DATA_COL])

    tf_idf_model = TfIdf()
    tf_idf_model.fit(train_tokenized)
    tf_idf_model_data = tf_idf_model.export()

    X_train = tf_idf_model.transform(train_tokenized)
    y_train_sentiment = np.where(cleaned_data[:, SENTIMENT_TARGET_COL] == POSITIVE, 1, -1)
    y_train_truthfulness = np.where(cleaned_data[:, TRUTHFULNESS_TARGET_COL] == TRUTHFUL, 1, -1)

    vanilla_perceptron_sentiment = VanillaPerceptron(
        max_iterations=1300,
        learning_rate=0.815,
        shuffle=True,
        score_func=partial(calculate_f1_score, average="macro"),
        rng=rng,
        debug=True,
        debug_at=50,
    )
    vanilla_perceptron_sentiment.fit(X_train, y_train_sentiment)
    vanilla_perceptron_sentiment_data = vanilla_perceptron_sentiment.export()

    vanilla_perceptron_truthfulness = VanillaPerceptron(
        max_iterations=834,
        learning_rate=20e-2,
        shuffle=True,
        score_func=partial(calculate_f1_score, average="macro"),
        rng=rng,
        debug=True,
        debug_at=50,
    )
    vanilla_perceptron_truthfulness.fit(X_train, y_train_truthfulness)
    vanilla_perceptron_truthfulness_data = vanilla_perceptron_truthfulness.export()


    vanilla_model_file_data = {
        "tf_idf_model": tf_idf_model_data,
        "sentiment_classifier": vanilla_perceptron_sentiment_data,
        "truthfulness_classifier": vanilla_perceptron_truthfulness_data,
    }

    averaged_perceptron_sentiment = AveragedPerceptron(
        max_iterations=699,
        learning_rate=3,
        shuffle=True,
        score_func=partial(calculate_f1_score, average="macro"),
        rng=rng,
        debug=True,
        debug_at=50,
    )
    averaged_perceptron_sentiment.fit(X_train, y_train_sentiment)
    averaged_perceptron_sentiment_data = averaged_perceptron_sentiment.export()

    averaged_perceptron_truthfulness = AveragedPerceptron(
        max_iterations=160,
        learning_rate=815e-2,
        shuffle=True,
        score_func=partial(calculate_f1_score, average="macro"),
        rng=rng,
        debug=True,
        debug_at=50,
    )
    averaged_perceptron_truthfulness.fit(
        X_train,
        y_train_truthfulness,
    )
    averaged_perceptron_truthfulness_data = averaged_perceptron_truthfulness.export()

    averaged_model_file_data = {
        "tf_idf_model": tf_idf_model_data,
        "sentiment_classifier": averaged_perceptron_sentiment_data,
        "truthfulness_classifier": averaged_perceptron_truthfulness_data,
    }

    store_model(VANILLA_MODEL_FILE_PATH, vanilla_model_file_data)
    store_model(AVERAGED_MODEL_FILE_PATH, averaged_model_file_data)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python perceplearn.py <input_file>")
        exit(1)
    input_file_path = sys.argv[1]
    main(input_file_path)
