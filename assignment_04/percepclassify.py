import sys
import argparse
from typing import Optional

import numpy as np

from constants import *
from data_cleaning import *
from perceptron import *
from tfidf import *
from utils import *


rng = np.random.default_rng(seed=RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def main(model_file_path: str, input_file_path: str, true_file_path: Optional[str] = None):

    data = load_data(input_file_path, type="DEV")

    # A dictionary containing the columns and a list of functions to perform on it in order
    data_cleaning_pipeline = {
        DEV_DATA_COL: [
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


    dev_tokenized = tokenize(cleaned_data[:, DEV_DATA_COL])

    # Load Model
    tf_idf_model_data, sentiment_model_data, truthfulness_model_data = load_model(
        model_file_path
    )

    tf_idf_saved_model = TfIdf()
    tf_idf_saved_model.load(tf_idf_model_data)

    X_dev = tf_idf_saved_model.transform(dev_tokenized)

    sentiment_model = None
    if sentiment_model_data["type"] == TYPE_VANILLA_PERPCETRON:
        sentiment_model = VanillaPerceptron()
    elif sentiment_model_data["type"] == TYPE_AVERAGED_PERCEPTRON:
        sentiment_model = AveragedPerceptron()
    sentiment_model.load(sentiment_model_data)

    truthfulness_model = None
    if truthfulness_model_data["type"] == TYPE_VANILLA_PERPCETRON:
        truthfulness_model = VanillaPerceptron()
    elif truthfulness_model_data["type"] == TYPE_AVERAGED_PERCEPTRON:
        truthfulness_model = AveragedPerceptron()
    truthfulness_model.load(truthfulness_model_data)

    # Give actual labels
    y_pred_sentiment = sentiment_model.predict(X_dev)
    y_pred_sentiment = np.where(y_pred_sentiment == -1, NEGATIVE, POSITIVE)

    y_pred_truthfulness = truthfulness_model.predict(X_dev)
    y_pred_truthfulness = np.where(y_pred_truthfulness == -1, DECEPTIVE, TRUTHFUL)

    if true_file_path:
        true_data = load_data(true_file_path, type="KEY")

        f1_sentiment = calculate_f1_score(true_data[:,SENTIMENT_TARGET_COL], y_pred_sentiment, average="macro")
        f1_truthful = calculate_f1_score(true_data[:,TRUTHFULNESS_TARGET_COL], y_pred_truthfulness, average="macro")

        print("Avg. F1: ", np.mean([f1_sentiment, f1_truthful]))

    output = list()
    for (id, truthfulness, sentiment) in zip(
        cleaned_data[:, DATA_ID_COL],
        y_pred_truthfulness,
        y_pred_sentiment,
    ):
        output.append((id, truthfulness, sentiment))

    store_predictions(OUTPUT_FILE_PATH, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file_path", type=str, help="the model file")
    parser.add_argument("input_file_path", type=str, help="the input file")
    parser.add_argument("--true", required=False, type=str, help="prints the f1 score for the input file")
    args = parser.parse_args()

    model_file_path = sys.argv[1]
    input_file_path = sys.argv[2]
    true_file_path = args.true if args.true else None

    main(model_file_path, input_file_path, true_file_path)
