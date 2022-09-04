#!/bin/python

# Python version used 3.10.6

from typing import List, Optional
import pandas as pd
import numpy as np
import nltk
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


DATA_PATH = "./data.tsv"
DATA_COL = "review_body"
TARGET_COL = "star_rating"
N_SAMPLES = 25000


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


def remove_html_and_url(data: pd.Series):
    """Function to remove
             1. HTML encodings
             2. HTML tags (both closed and open)
             3. URLs

    Args:
        data (pd.Series): A Pandas series of type string

    Returns:
        _type_: pd.Series
    """
    # Remove HTML encodings
    data.str.replace(r"&#\d+;", " ", regex=True)

    # Remove HTML tags (both open and closed)
    data.str.replace(r"<[a-zA-Z]+\s?/?>", " ", regex=True)

    # Remove URLs
    data.str.replace(r"https?://([\w\-\._]+){2,}/[\w\-\.\-/=\+_\?]+", " ", regex=True)

    return data


# Remove non-alphabetical characters
def remove_non_alpa_characters(data: pd.Series):
    return data.str.replace(r"_+|\\|[^a-zA-Z\s]", " ", regex=True)


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
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    accuracy_score = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    print(f"{accuracy_score}")

    for rating_precision, rating_recall, rating_f1 in zip(precision, recall, f1):
        print(f"{rating_precision},{rating_recall},{rating_f1}")

    print(f"{np.mean(precision)},{np.mean(recall)},{np.mean(f1)}")


def load_data(file_path: str, sep: str, cols: Optional[List[str]]):
    return pd.read_csv(file_path, sep=sep, usecols=cols)


##############################################################################
### Main Function
##############################################################################
def main():

    data = load_data(DATA_PATH, "\t", [DATA_COL, TARGET_COL])

    # Drop the outlier which is star_rating = "2012-12-21"
    data = data[data[DATA_COL] != "2012-12-21"]

    # Remove nan valued rows
    data = data[data[TARGET_COL].notnull()]
    data = data[data[DATA_COL].notnull()]

    # Convert all star rating to integer
    data[DATA_COL] = data[DATA_COL].astype(int)

    ##############################################################################
    ### Sample data
    ##############################################################################
    sampled_data = data.groupby(TARGET_COL, group_keys=False).apply(lambda x: x.sample(N_SAMPLES))

    ##############################################################################
    ### Cleaning
    ##############################################################################

    avg_len_before_cleaning = sampled_data.review_body.str.len().mean()
    data_cleaning_pipeline = {
        "review_body": [
            to_lower,
            remove_accented_characters,
            remove_html_and_url,
            fix_contractions,
            remove_non_alpa_characters,
            remove_extra_spaces,
        ]
    }

    cleaned_data = sampled_data.copy()

    # Process all the cleaning instructions
    for col, pipeline in data_cleaning_pipeline.items():
        # Get the column to perform cleaning on
        temp_data = data[col]

        # Perform all the cleaning functions sequencially
        for func in pipeline:
            temp_data = func(temp_data)

        # Replace the old column with cleaned one.
        cleaned_data[col] = temp_data

    avg_len_after_cleaning = cleaned_data.review_body.str.len().mean()

    print(f"{avg_len_before_cleaning,avg_len_after_cleaning}")

    #####################################################################################
    ### Preprocessing
    #####################################################################################

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
            if func.__name__ == "wordnet_lemmatizer":
                temp_data = func(temp_data, consider_pos_tag=False)
            else:
                temp_data = func(temp_data)

        # Replace the old column with cleaned one.
        preprocessed_data[col] = temp_data.copy()

    avg_len_after_preprocessing = preprocessed_data[DATA_COL].str.len().mean()

    print(f"{avg_len_before_preprocessing},{avg_len_after_preprocessing}")

    # Drop empty strings
    preprocessed_data["review_body"].replace("", np.nan, inplace=True)
    preprocessed_data.dropna(subset=["review_body"], inplace=True)

    # Split
    data = preprocessed_data.groupby("star_rating", group_keys=False).apply(lambda x: x.sample(20000))
    train, test = train_test_split(data, test_size=0.2, stratify=data["star_rating"])

    #####################################################################################
    ### Feature Extraction
    #####################################################################################

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    vectorizer.fit(train["review_body"])

    X_tfidf_train = vectorizer.transform(train["review_body"])
    X_tfidf_test = vectorizer.transform(test["review_body"])
    y_train = train["star_rating"]
    y_test = test["star_rating"]

    #####################################################################################
    ### Models
    #####################################################################################
    # 1. Perceptron
    perceptron_clf = Perceptron()
    perceptron_clf.fit(X_tfidf_train, y_train)

    y_pred = perceptron_clf.predict(X_tfidf_test)

    calc_metrics(y_test, y_pred)

    # 2. Logistic Regression
    perceptron_clf = LogisticRegression()
    perceptron_clf.fit(X_tfidf_train, y_train)

    y_pred = perceptron_clf.predict(X_tfidf_test)

    calc_metrics(y_test, y_pred)

    # 4. SVM
    svm_clf = LinearSVC()
    svm_clf.fit(X_tfidf_train, y_train)

    y_pred = svm_clf.predict(X_tfidf_test)

    calc_metrics(y_test, y_pred)

    # 5. Naive Bayes
    nb_clf = MultinomialNB()
    nb_clf.fit(X_tfidf_train, y_train)

    y_pred = nb_clf.predict(X_tfidf_test)

    calc_metrics(y_test, y_pred)


if __name__ == "__main__":
    main()
