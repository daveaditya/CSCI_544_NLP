#!/bin/python3

import sys
import csv
import numpy as np
from typing import Any, Dict, List


###################################################################
### Constants
###################################################################


DATA_PATH = "./data"
DATASET_FILES = {
    "ITALIAN": {
        "TRAIN": f"{DATA_PATH}/it_isdt_train_tagged.txt",
        "DEV": {
            "RAW": f"{DATA_PATH}/it_isdt_dev_raw.txt",
            "TAGGED": f"{DATA_PATH}/it_isdt_train_tagged.txt",
        },
    },
    "JAPANESE": {
        "TRAIN": f"{DATA_PATH}/ja_gsd_train_tagged.txt",
        "DEV_RAW": f"{DATA_PATH}/ja_gsd_dev_raw.txt",
        "DEV_TAGGED": f"{DATA_PATH}/ja_gsd_train_tagged.txt",
    },
}
MODEL_FILE = "hmmmodel.txt"


###################################################################
### Helper Functions
###################################################################


def load_training_data(input_file: str) -> List[str]:
    pass


def load_test_data(input_file: str) -> List[str]:
    pass


def save_model(model: Any, model_file: str):
    pass


###################################################################
### Main Program
###################################################################
def main(input_file: str, mode: str = "dev"):

    # Load training data

    # Train

    # Store model

    pass


if __name__ == "__main__":
    input_file = "ITALIAN"
    mode = "dev"
    if len(sys.argv) == 2:
        input_file = sys.argv[1]
        mode = "prod"
    main(input_file, mode = "prod")
