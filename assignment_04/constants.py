######################################################
### Constants                                      ###
######################################################
# Base Paths
INPUT_PATH = "./data"
MODEL_PATH = "./model"
OUTPUT_PATH = "./output"

# Model File names
VANILLA_MODEL_FILENAME = "vanillamodel.txt"
AVERAGED_MODEL_FILENAME = "averagedmodel.txt"
OUTPUT_FILENAME = "output.txt"

# Class Identifiers
TRUTHFUL = "True"
DECEPTIVE = "Fake"
POSITIVE = "Pos"
NEGATIVE = "Neg"

TYPE_VANILLA_PERPCETRON = "vanilla_perceptron"
TYPE_AVERAGED_PERCEPTRON = "averaged_perceptron"

# File paths
TRAIN_FILE_PATH = f"{INPUT_PATH}/train-labeled.txt"
CLEANED_DATA_FILE_PATH = f"{INPUT_PATH}/cleaned-data.txt"
PREPROCESSED_DATA_FILE_PATH = f"{INPUT_PATH}/preprocessed-data.txt"

VANILLA_MODEL_FILE_PATH = f"{MODEL_PATH}/{VANILLA_MODEL_FILENAME}"
AVERAGED_MODEL_FILE_PATH = f"{MODEL_PATH}/{AVERAGED_MODEL_FILENAME}"

OUTPUT_FILE_PATH = f"{OUTPUT_PATH}/{OUTPUT_FILENAME}"

DEV_DATA_FILE_PATH = f"{INPUT_PATH}/dev-text.txt"
DEV_KEY_FILE_PATH = f"{INPUT_PATH}/dev-key.txt"

RANDOM_SEED = 42

DATA_ID_COL = 0
TRAIN_DATA_COL = 3
DEV_DATA_COL = 1
SENTIMENT_TARGET_COL = 2
TRUTHFULNESS_TARGET_COL = 1
TEST_SIZE = 0.2
