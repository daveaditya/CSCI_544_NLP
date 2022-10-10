#!/bin/python3

import sys
import csv
import json

import numpy as np


###################################################################
### Constants
###################################################################


DATA_PATH = "./data"
OUTPUT_PATH = "./submission"
DATASET_FILES = {
    "ITALIAN": {
        "TRAIN": f"{DATA_PATH}/it_isdt_train_tagged.txt",
        "DEV_RAW": f"{DATA_PATH}/it_isdt_dev_raw.txt",
        "DEV_TAGGED": f"{DATA_PATH}/it_isdt_dev_tagged.txt",
    },
    "JAPANESE": {
        "TRAIN": f"{DATA_PATH}/ja_gsd_train_tagged.txt",
        "DEV_RAW": f"{DATA_PATH}/ja_gsd_dev_raw.txt",
        "DEV_TAGGED": f"{DATA_PATH}/ja_gsd_dev_tagged.txt",
    },
}
MODEL_FILE = f"{OUTPUT_PATH}/hmmmodel.txt"
OUTPUT_FILE = f"{OUTPUT_PATH}/hmmoutput.txt"

START_TAG = "<ST@RT$>"

SMOOTHING_PARAMETER = 1.0


###################################################################
### Helper Functions
###################################################################


def load_document(file_path: str):
    document = list()
    with open(file_path, mode="r") as file:
        csv_reader = csv.reader(file, delimiter=" ", skipinitialspace=True, quoting=csv.QUOTE_NONE)
        for sentence in csv_reader:
            document.append(sentence)
    return document


def write_output(output_file_path: str, predicteds_tags: str):
    with open(output_file_path, mode="w") as file:
        for predicted_row in predicteds_tags:
            file.write(" ".join(predicted_row) + "\n")


def load_model(model_path: str):
    """Load the model file to respective objects
    """
    model_data = None
    with open(model_path, mode="r") as model_file:
        model_data = json.load(model_file)
    return (
        model_data["words"],
        model_data["tags"],
        model_data["open_class_tags"],
        model_data["tag_counts"],
        model_data["smoothing_parameter"],
        np.array(model_data["transition_probabilities"]),
        model_data["transition_matrix_labels"],
        np.array(model_data["emission_probabilities"]),
        model_data["emission_matrix_row_labels"],
        model_data["emission_matrix_col_labels"],
    )


def accuracy(tagged_true, tagged_preds):
    total_count, correct_count = 0, 0
    for sentence_true, sentence_pred in zip(tagged_true, tagged_preds):
        for word_tag_true, word_tag_pred in zip(sentence_true, sentence_pred):
            if word_tag_true == word_tag_pred:
                correct_count += 1
            total_count += 1
    return correct_count / total_count


###################################################################
### Viterbi Algorithm
###################################################################


def viterbi_decoding(
    tags,
    tag_counts,
    open_class_tags,
    emission_probabilities,
    emission_matrix_row_labels,
    emission_matrix_col_labels,
    transition_probabilities,
    transition_matrix_labels,
    sentence,
    smoothing_parameter,
):
    n_words_in_sentence = len(sentence)
    n_tags = len(tags)

    viterbi_matrix = np.zeros(shape=(n_tags, n_words_in_sentence), dtype=np.float64)
    backtrack_matrix = np.zeros(shape=(n_tags, n_words_in_sentence), dtype=np.int32)

    cumulative_probability = 0

    for idx, tag in enumerate(tags):
        # handle new word in corpus
        word = sentence[0]

        # Emission Probablity
        # approach: set emission probability = 1 i.e. use transision probability alone
        if word not in emission_matrix_row_labels:
            em_prob = 1.0

        # as word is already checked, it is already there in emission matrix, just need to check if a corresponding tag exists or not
        elif tag not in emission_matrix_col_labels:
            em_prob = 0.0

        else:
            em_prob = emission_probabilities[emission_matrix_row_labels[word]][emission_matrix_col_labels[tag]]

        # Transision Probability
        trans_prob = transition_probabilities[transition_matrix_labels[START_TAG]][transition_matrix_labels[tag]]
        if trans_prob == -1.0:
            trans_prob = float(1 / (tag_counts[START_TAG] + n_tags))

        viterbi_matrix[idx][0] = trans_prob * em_prob

        backtrack_matrix[idx][0] = 0

    for idx in range(1, n_words_in_sentence):

        word = sentence[idx]
        is_new_word = word not in emission_matrix_row_labels

        for end_tag in tags:

            for start_tag in tags:

                # emission
                if is_new_word:
                    em_prob = 1.0
                elif end_tag not in emission_matrix_col_labels:
                    em_prob = 0.0
                else:
                    em_prob = emission_probabilities[emission_matrix_row_labels[word]][
                        emission_matrix_col_labels[end_tag]
                    ]
                    if em_prob == 0.0:
                        continue

                trans_prob = transition_probabilities[transition_matrix_labels[start_tag]][
                    transition_matrix_labels[end_tag]
                ]
                if trans_prob == 0:
                    continue
                elif trans_prob == -1.0:
                    trans_prob = smoothing_parameter / (tag_counts[start_tag] + smoothing_parameter * n_tags)

                cumulative_probability = (
                    viterbi_matrix[transition_matrix_labels[start_tag]][idx - 1] * trans_prob * em_prob
                )
                if cumulative_probability == 0:
                    continue

                if cumulative_probability > viterbi_matrix[transition_matrix_labels[end_tag]][idx]:
                    viterbi_matrix[transition_matrix_labels[end_tag]][idx] = cumulative_probability
                    backtrack_matrix[transition_matrix_labels[end_tag]][idx] = transition_matrix_labels[start_tag]
                else:
                    continue

    return (viterbi_matrix, backtrack_matrix)


def viterbi_backtrack(tags, viterbi_matrix, backtrack_matrix, sentence):
    n_tags = len(tags)
    n_words_in_sentence = len(sentence)

    # Backtracking
    best_idx = 0
    for i in range(n_tags):
        if viterbi_matrix[i][n_words_in_sentence - 1] > viterbi_matrix[best_idx][n_words_in_sentence - 1]:
            best_idx = i

    output = [f"{sentence[n_words_in_sentence - 1]}/{tags[best_idx]}"]

    for idx in range(n_words_in_sentence - 1, 0, -1):
        best_idx = backtrack_matrix[best_idx][idx]
        output.insert(0, f"{sentence[idx - 1]}/{tags[best_idx]}")

    return output


###################################################################
### Main Program
###################################################################
def main(test_file: str, show_accuracy: bool = False, lang: str = None):
    # Load Model
    (
        words,
        tags,
        open_class_tags,
        tag_counts,
        SMOOTHING_PARAMETER,
        transition_probabilities,
        transition_matrix_labels,
        emission_probabilities,
        emission_matrix_row_labels,
        emission_matrix_col_labels,
    ) = load_model(MODEL_FILE)


    # Load test file
    test_document = load_document(test_file)

    # Make predictions
    predicted_tags = list()
    for sentence in test_document:
        viterbi_matrix, backtrack_matrix = viterbi_decoding(
            tags,
            tag_counts,
            open_class_tags,
            emission_probabilities,
            emission_matrix_row_labels,
            emission_matrix_col_labels,
            transition_probabilities,
            transition_matrix_labels,
            sentence,
            SMOOTHING_PARAMETER,
        )
        output = viterbi_backtrack(tags, viterbi_matrix, backtrack_matrix, sentence)
        predicted_tags.append(output)


    # Store predictions
    write_output(OUTPUT_FILE, predicted_tags)

    # Print accuracy if needed
    if show_accuracy:
        tagged_true = (
            load_document(DATASET_FILES["ITALIAN"]["DEV_TAGGED"])
            if lang == "ITALIAN"
            else load_document(DATASET_FILES["JAPANESE"]["DEV_TAGGED"])
        )

        print(accuracy(tagged_true, predicted_tags))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hmmlearn.py")
        exit(1)

    show_accuracy = False
    test_file = sys.argv[1]
    lang = None

    if "ITALIAN" == test_file:
        test_file = DATASET_FILES["ITALIAN"]["DEV_RAW"]
        show_accuracy = True
        lang = "ITALIAN"
    elif "JAPANESE" == test_file:
        test_file = DATASET_FILES["JAPANESE"]["DEV_RAW"]
        show_accuracy = True
        lang = "JAPANESE"

    main(test_file, show_accuracy, lang)
