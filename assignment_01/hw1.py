#!/bin/python

# Python version used 3.10.6

from typing import List, Optional, Set
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
import re


##############################################################################
### Defining Constants
##############################################################################

DATA_PATH = "./"
DATA_FILE = "data.tsv"

DATA_COL = "review_body"
TARGET_COL = "star_rating"
N_SAMPLES = 20000

RANDOM_SEED = 42


##############################################################################
### Data Cleaning Functions
##############################################################################
# Convert all reviews to lower case (optional according to study)
def to_lower(data: pd.Series):
    return data.str.lower()


def remove_accented_characters(data: pd.Series):
    import unicodedata

    """Removes accented characters from the Series

    Args:
        data (pd.Series): Series of string

    Returns:
        _type_: pd.Series
    """
    import unicodedata

    return data.apply(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("utf-8", "ignore"))


# Remove HTML encodings
def remove_html_encodings(data: pd.Series):
    return data.str.replace(r"&#\d+;", " ", regex=True)


# Remove HTML tags (both open and closed)
def remove_html_tags(data: pd.Series):
    return data.str.replace(r"<[a-zA-Z]+\s?/?>", " ", regex=True)


# Remove URLs
def remove_url(data: pd.Series):
    return data.str.replace(r"https?://([\w\-\._]+){2,}/[\w\-\.\-/=\+_\?]+", " ", regex=True)


# Handle emoji
def convert_emoji_to_txt(data: pd.Series):
    from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO

    EMO_TO_TXT_DICT = dict()
    for emot in UNICODE_EMOJI:
        EMO_TO_TXT_DICT[emot] = f" {re.sub(r',|:|_', '', UNICODE_EMOJI[emot])} "

    for emo in EMOTICONS_EMO:
        EMO_TO_TXT_DICT[emot] = f" {re.sub(r',| ', '', EMOTICONS_EMO[emo])} "

    def convert_emojis(text, emo_to_txt_dict):
        for emot in emo_to_txt_dict:
            text = text.replace(emot, emo_to_txt_dict[emot])
        return text

    return data.apply(lambda x: convert_emojis(x, EMO_TO_TXT_DICT))


# Replaces numbers with NUM tag
def replace_digits_with_tag(data: pd.Series):
    return data.str.replace(r"\d+", " NUM ", regex=True)


# Remove non-alphabetical characters
def remove_non_alpha_characters(data: pd.Series):
    return data.str.replace(r"_+|\\|[^a-zA-Z0-9\s]", " ", regex=True)


# Remove extra spaces
def remove_extra_spaces(data: pd.Series):
    return data.str.replace(r"^\s*|\s\s*", " ", regex=True)


# Expanding contractions
def fix_contractions(data: pd.Series):
    import contractions

    def contraction_fixer(txt: str):
        return " ".join([contractions.fix(word) for word in txt.split()])

    return data.apply(contraction_fixer)


##############################################################################
### Data Preprocessing Functions
##############################################################################
def tokenize(data: pd.Series):
    from nltk.tokenize import word_tokenize

    nltk.download("punkt")

    return data.apply(word_tokenize)


def remove_stopwords(data: pd.Series):
    """Remove stop words using the NLTK stopwords dictionary

    Args:
        string (str): a document

    Returns:
        str: a document with stopwords removed
    """
    from nltk.corpus import stopwords

    nltk.download("stopwords")

    stopwords = set(stopwords.words())

    def remover(word_list: List[str], stopwords: Set[str]):
        return [word for word in word_list if not word in stopwords]

    return data.apply(lambda word_list: remover(word_list, stopwords))


def lemmatize(data: pd.Series, consider_pos_tag: bool = True):
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer

    nltk.download("omw-1.4")

    # POS tagging
    def perform_nltk_pos_tag(data: pd.Series):
        from nltk import pos_tag

        nltk.download("averaged_perceptron_tagger")

        return data.apply(pos_tag)

    # Convert POS tag to wordnet pos tags
    def wordnet_pos_tagger(tag: str):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return None

    lemmatizer = WordNetLemmatizer()
    lemmatized = list()

    if consider_pos_tag:
        pos_tagged_data = data.copy()
        pos_tagged_data = perform_nltk_pos_tag(data)

        for row in pos_tagged_data:

            lemmatized_row = list()

            if consider_pos_tag:
                for word, tag in row:
                    wordnet_pos_tag = wordnet_pos_tagger(tag)

                    if wordnet_pos_tag is None:
                        lemmatized_row.append(word)
                    else:
                        result = lemmatizer.lemmatize(word, wordnet_pos_tag)
                        lemmatized_row.append(lemmatizer.lemmatize(word, wordnet_pos_tag))

            lemmatized.append(lemmatized_row)
    else:
        for row in data:
            lemmatized_row = list()

            for word in row:
                lemmatized_row.append(lemmatizer.lemmatize(word))

            lemmatized.append(lemmatized_row)

    return pd.Series(lemmatized)


# Concatenate lemmatized sentences back into one sentence
def concatenate(data: pd.Series):
    return data.apply(lambda words: " ".join(words))


##############################################################################
### Helper Functions
##############################################################################
def calc_metrics(y_true, y_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    for rating_precision, rating_recall, rating_f1 in zip(precision, recall, f1):
        print(f"{rating_precision},{rating_recall},{rating_f1}")

    print(f"{np.mean(precision)},{np.mean(recall)},{np.mean(f1)}")


def load_data(file_path: str, sep: str, cols: Optional[List[str]]):
    return pd.read_csv(file_path, sep=sep, usecols=cols, low_memory=True)


##############################################################################
### Main Function
##############################################################################
def main():

    data = load_data(f"{DATA_PATH}/{DATA_FILE}", sep="\t", cols=[TARGET_COL, DATA_COL])

    # Drop NA
    data.dropna(inplace=True)

    # Remove nan valued rows
    data = data[data[TARGET_COL].notnull()].copy()
    data = data[data[DATA_COL].notnull()].copy()

    # Drop the outlier which is star_rating = "2012-12-21"
    data = data[data[DATA_COL] != "2012-12-21"].copy()

    # Convert all star rating to integer
    data[TARGET_COL] = data[TARGET_COL].astype(int)

    ##############################################################################
    ### Sample data
    ##############################################################################
    sampled_data = data.groupby(TARGET_COL, group_keys=False).apply(
        lambda x: x.sample(N_SAMPLES, random_state=RANDOM_SEED)
    )
    sampled_data.reset_index(inplace=True)
    sampled_data.drop(columns=["index"], inplace=True)

    ##############################################################################
    ### Cleaning
    ##############################################################################

    avg_len_before_cleaning = sampled_data.review_body.str.len().mean()

    # A dictionary containing the columns and a list of functions to perform on it in order
    data_cleaning_pipeline = {
        DATA_COL: [
            convert_emoji_to_txt,
            to_lower,
            remove_accented_characters,
            remove_html_encodings,
            remove_html_tags,
            remove_url,
            fix_contractions,
            remove_non_alpha_characters,
            remove_extra_spaces,
        ]
    }

    cleaned_data = sampled_data.copy()

    # Process all the cleaning instructions
    for col, pipeline in data_cleaning_pipeline.items():
        # Get the column to perform cleaning on
        temp_data = cleaned_data[col].copy()

        # Perform all the cleaning functions sequencially
        for func in pipeline:
            temp_data = func(temp_data)

        # Replace the old column with cleaned one.
        cleaned_data[col] = temp_data.copy()

    avg_len_after_cleaning = cleaned_data.review_body.str.len().mean()

    print(f"{avg_len_before_cleaning},{avg_len_after_cleaning}")

    #####################################################################################
    ### Preprocessing
    ###################################################################################

    avg_len_before_preprocessing = cleaned_data[DATA_COL].str.len().mean()

    preprocessing_pipeline = {DATA_COL: [tokenize, lemmatize, concatenate]}

    # Run the pipeline
    preprocessed_data = cleaned_data.copy()

    # Process all the cleaning instructions
    for col, pipeline in preprocessing_pipeline.items():
        # Get the column to perform cleaning on
        temp_data = preprocessed_data[col].copy()

        # Perform all the cleaning functions sequencially
        for func in pipeline:

            if func.__name__ == "lemmatize":
                temp_data = func(temp_data, consider_pos_tag=True)
            else:
                temp_data = func(temp_data)

        # Replace the old column with cleaned one.
        preprocessed_data[col] = temp_data.copy()

    avg_len_after_preprocessing = preprocessed_data[DATA_COL].str.len().mean()

    print(f"{avg_len_before_preprocessing},{avg_len_after_preprocessing}")

    #####################################################################################
    ### Feature Extraction
    ###################################################################################

    # Drop empty strings
    preprocessed_data = preprocessed_data[preprocessed_data[DATA_COL].str.len() != 0]

    final_data = preprocessed_data.copy()

    # Split the data 80-20 split
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(final_data, test_size=0.2, stratify=final_data[TARGET_COL], random_state=RANDOM_SEED)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.tokenize import word_tokenize

    nltk.download("punkt")

    vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
    vectorizer.fit(final_data[DATA_COL])

    X_tfidf_train = vectorizer.transform(train[DATA_COL])
    X_tfidf_test = vectorizer.transform(test[DATA_COL])
    y_train = train[TARGET_COL]
    y_test = test[TARGET_COL]

    #####################################################################################
    ### Models
    ###################################################################################

    # 1. Perceptron
    from sklearn.linear_model import Perceptron

    class_weight = {1: 0.9525, 2: 1.99825, 3: 1.9225, 4: 0.625, 5: 0.8585}
    clf = Perceptron(
        max_iter=8000, alpha=0.012, random_state=RANDOM_SEED, tol=1e-4, early_stopping=True, class_weight="balanced"
    )  # 0.45103975891511683

    clf.fit(X_tfidf_train, y_train)

    y_pred = clf.predict(X_tfidf_test)

    calc_metrics(y_test, y_pred)

    # 2. SVM
    from sklearn.svm import LinearSVC

    class_weight = {1: 0.9525, 2: 1.99825, 3: 1.9225, 4: 0.625, 5: 0.8585}

    clf = LinearSVC(dual=False, C=0.1, max_iter=1000, class_weight=class_weight, random_state=RANDOM_SEED)

    clf.fit(X_tfidf_train, y_train)

    y_pred = clf.predict(X_tfidf_test)

    calc_metrics(y_test, y_pred)

    # 3. Logistic Regression
    from sklearn.linear_model import LogisticRegression

    class_weight = {1: 0.9525, 2: 1.99825, 3: 1.9225, 4: 0.625, 5: 0.8585}

    clf = LogisticRegression(
        penalty="l2",
        solver="saga",
        max_iter=200,
        multi_class="multinomial",
        C=0.1024,
        random_state=RANDOM_SEED,
        class_weight=class_weight,
    )

    clf.fit(X_tfidf_train, y_train)

    y_pred = clf.predict(X_tfidf_test)

    calc_metrics(y_test, y_pred)

    # 4. Naive Bayes
    from sklearn.naive_bayes import MultinomialNB

    class_prior = [0.2, 0.2048, 0.2, 0.2, 0.22476]

    clf = MultinomialNB(alpha=32.8824, class_prior=class_prior)

    clf.fit(X_tfidf_train, y_train)

    y_pred = clf.predict(X_tfidf_test)

    calc_metrics(y_test, y_pred)


if __name__ == "__main__":
    main()
