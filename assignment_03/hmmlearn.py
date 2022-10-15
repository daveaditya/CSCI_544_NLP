#!/bin/python3

import sys
import csv
import json
from copy import deepcopy
from typing import List, Dict

import numpy as np


###################################################################
### Constants
###################################################################


DATA_PATH = "./data"
MODEL_PATH = "./model"
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
MODEL_FILE = f"{MODEL_PATH}/hmmmodel.txt"

START_TAG = "<ST@RT$>"

SMOOTHING_PARAMETER = 1.0
OPEN_CLASS_PRECENT = 1.0

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



def write_model(
    out_file_path: str,
    words: List[str],
    tags: List[str],
    open_class_tags: List[str],
    tag_counts: Dict[str, int],
    transition_probabilities,
    transition_matrix_labels,
    emission_probabilities,
    emission_matrix_row_labels,
    emission_matrix_col_labels,
    smoothing_parameter,
):
    """Writes the model parameters to a txt file in JSON format

    Args:
        out_file_path (str): output model path
        words (List[str]): list of words
        tags (List[str]): list of tags
        open_class_tags (List[str]): list of open class tags
        tag_counts (Dict[str, int]): list of tag counts
        transition_probabilities (_type_): list of transition probabilities
        transition_matrix_labels (_type_): list of transition matric labels
        emission_probabilities (_type_): list of emission probabilities
        emission_matrix_row_labels (_type_): list of emission matrix row labels
        emission_matrix_col_labels (_type_): list of emission matrix column labels
        smoothing_parameter (_type_): smoothing parameter for laplace smoothing
    """
    with open(out_file_path, mode="w") as output_file:
        out = dict()
        out["tags"] = tags
        out["open_class_tags"] = open_class_tags
        out["words"] = words
        out["tag_counts"] = tag_counts
        out["smoothing_parameter"] = smoothing_parameter
        out["transition_probabilities"] = transition_probabilities.tolist()
        out["transition_matrix_labels"] = transition_matrix_labels
        out["emission_probabilities"] = emission_probabilities.tolist()
        out["emission_matrix_row_labels"] = emission_matrix_row_labels
        out["emission_matrix_col_labels"] = emission_matrix_col_labels
        json.dump(out, output_file, ensure_ascii=False)



###################################################################
### Emission and Transition Probalities
###################################################################

def count_occurrences(train_document: List[List[str]]):
    tag_counts = {
        START_TAG: len(train_document),
    }
    word_tag_counts = {}
    tag_tag_counts = {
        START_TAG: {},
    }

    count = len(train_document)

    # Process count number of sentences from document
    for idx, sentence in enumerate(train_document):
        if idx == count:
            break

        prev_tag = START_TAG
        sentence_last_idx = len(sentence) - 1
        for idx, word_tag_pair in enumerate(sentence):
            # Extract word tag
            word, tag = word_tag_pair.rsplit("/", 1)

            # Count the Tag!
            if tag not in tag_counts:
                tag_counts[tag] = 1
            else:
                tag_counts[tag] += 1

            # Count the Word - Tag (Emission)
            if word not in word_tag_counts:
                word_tag_counts[word] = {tag: 1}
            else:
                # Check if the tag is in the dict
                if tag not in word_tag_counts[word]:
                    word_tag_counts[word][tag] = 1
                else:
                    word_tag_counts[word][tag] += 1

            # Count tag-tag (Transition)
            if prev_tag in tag_tag_counts:
                if tag not in tag_tag_counts[prev_tag]:
                    tag_tag_counts[prev_tag][tag] = 1
                else:
                    tag_tag_counts[prev_tag][tag] += 1
            else:
                tag_tag_counts[prev_tag] = {tag: 1}

            prev_tag = tag

    return (tag_counts, tag_tag_counts, word_tag_counts)


def calculate_probabilities(
    tags: List[str],
    words: List[str],
    tag_counts: Dict[str, int],
    tag_tag_counts: Dict[str, Dict[str, int]],
    word_tag_counts: Dict[str, Dict[str, int]],
    smoothing_parameter: float,
):
    # Create row and column headers for access
    # Transition Matric Labels (same for both row and column)
    transition_matrix_labels = {tag: i for i, tag in enumerate(tags)}
    transition_matrix_n_rows, transition_matrix_n_cols = len(transition_matrix_labels), len(transition_matrix_labels)

    # Emission Matrix Labels
    emission_col_labels = deepcopy(tags)
    emission_col_labels.remove(START_TAG)

    emission_matrix_n_rows, emission_matrix_n_cols = len(words), len(emission_col_labels)
    emission_matrix_row_labels = {word: i for i, word in enumerate(words)}
    emission_matrix_col_labels = {tag: i for i, tag in enumerate(emission_col_labels)}

    # Create empty transition and emission probability matrices
    transition_probabilities = np.zeros(shape=(transition_matrix_n_rows, transition_matrix_n_cols), dtype=np.float64)
    emission_probabilities = np.zeros(shape=(emission_matrix_n_rows, emission_matrix_n_cols), dtype=np.float64)

    # Fill in emission probablity matrix
    for row_word, row_idx in emission_matrix_row_labels.items():
        for col_tag, col_idx in emission_matrix_col_labels.items():
            if col_tag not in word_tag_counts[row_word]:
                emission_probabilities[row_idx][col_idx] = 0.0
            else:
                emission_probability = word_tag_counts[row_word][col_tag] / tag_counts[col_tag]

                if emission_probability > 1:
                    emission_probability = 1

                emission_probabilities[row_idx][col_idx] = emission_probability

    # Fill in transition probablity matrix
    for row_tag, row_idx in transition_matrix_labels.items():
        for col_tag, col_idx in transition_matrix_labels.items():
            if col_tag not in tag_tag_counts[row_tag]:
                transition_probabilities[row_idx][col_idx] = -1.0
            else:
                # Laplace Smoothing
                transition_probabilities[row_idx][col_idx] = (
                    tag_tag_counts[row_tag][col_tag] + smoothing_parameter
                ) / (tag_counts[row_tag] + smoothing_parameter * len(tag_counts))

    return (
        transition_probabilities,
        transition_matrix_labels,
        emission_probabilities,
        emission_matrix_row_labels,
        emission_matrix_col_labels,
    )


def calculate_open_classes(
    emission_probabilities, tags, threshold: float = 0.2
):
    n_open_tags = int(threshold * len(tags))
    
    unqiue_counts = (emission_probabilities != 0).sum(axis=0)

    reverse_sorted_counts = unqiue_counts.argsort()[::-1]
    open_class_tags_idx = reverse_sorted_counts[:n_open_tags]
    open_class_tags = list(map(tags.__getitem__, open_class_tags_idx))

    return open_class_tags


###################################################################
### Main Program
###################################################################
def main(input_file: str):

    # Load training data
    train_document = load_document(input_file)

    # Train
    # Count occurences of tags, tag followed by tag and tag for word
    tag_counts, tag_tag_counts, word_tag_counts = count_occurrences(train_document)

    # Get list of all words and tags
    words = list(word_tag_counts.keys())
    tags = list(tag_counts.keys())

    # Calculate Transition and Emission Probabilities
    (
        transition_probabilities,
        transition_matrix_labels,
        emission_probabilities,
        emission_matrix_row_labels,
        emission_matrix_col_labels,
    ) = calculate_probabilities(tags, words, tag_counts, tag_tag_counts, word_tag_counts, SMOOTHING_PARAMETER)

    open_class_tags = calculate_open_classes(emission_probabilities, tags, OPEN_CLASS_PRECENT)

    # Save the model
    write_model(
        MODEL_FILE,
        words,
        tags,
        open_class_tags,
        tag_counts,
        transition_probabilities,
        transition_matrix_labels,
        emission_probabilities,
        emission_matrix_row_labels,
        emission_matrix_col_labels,
        SMOOTHING_PARAMETER,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hmmlearn.py")
        exit(1)

    input_file = sys.argv[1]

    if "ITALIAN" == input_file:
        input_file = DATASET_FILES["ITALIAN"]["TRAIN"]
    elif "JAPANESE" == input_file:
        input_file = DATASET_FILES["JAPANESE"]["TRAIN"]

    main(input_file)
