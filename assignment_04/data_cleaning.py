import numpy as np
import numpy.typing as npt

from contractions import fix as contraction_fix

# Convert all reviews to lower case (optional according to study)
def to_lower(data: npt.NDArray):
    n_data = data.shape[0]
    result = data.copy()
    for i in range(n_data):
        result[i] = result[i].lower()
    return result


def remove_html_encodings(data: npt.NDArray):
    import re

    n_data = data.shape[0]
    result = data.copy()
    for i in range(n_data):
        result[i] = re.sub(r"&#\d+;", " ", result[i])
    return result


def remove_html_tags(data: npt.NDArray):
    import re

    n_data = data.shape[0]
    result = data.copy()
    for i in range(n_data):
        result[i] = re.sub(r"<[a-zA-Z]+\s?/?>", "", result[i])
    return result


def remove_url(data: npt.NDArray):
    import re

    n_data = data.shape[0]
    result = data.copy()
    for i in range(n_data):
        result[i] = re.sub(r"https?://([\w\-\._]+){2,}/[\w\-\.\-/=\+_\?]+", "", result[i])
    return result


def remove_html_and_url(data):
    """Function to remove
             1. HTML encodings
             2. HTML tags (both closed and open)
             3. URLs

    Args:
        data (npt.NDArray): A Numpy Array of type string

    Returns:
        _type_: npt.NDArray
    """
    import re

    n_data = data.shape[0]
    result = data.copy()
    for i in range(n_data):
        # Remove HTML encodings
        result[i] = re.sub(r"&#\d+;", "", result[i])

        # Remove HTML tags (both open and closed)
        result[i] = re.sub(r"<[a-zA-Z]+\s?/?>", "", result[i])

        # Remove URLs
        result[i] = re.sub(r"https?://([\w\-\._]+){2,}/[\w\-\.\-/=\+_\?]+", "", result[i])

    return result


def replace_digits_with_tag(data: npt.NDArray):
    import re

    n_data = data.shape[0]
    result = data.copy()
    for i in range(n_data):
        result[i] = re.sub(r"\d+", " NUM ", result[i])
    return result


# Remove non-alphabetical characters
def remove_non_alpha_characters(data: npt.NDArray):
    import re

    n_data = data.shape[0]
    result = data.copy()
    for i in range(n_data):
        result[i] = re.sub(r"_+|\\|[^a-zA-Z0-9\s]", " ", result[i])
    return result


# Remove extra spaces
def remove_extra_spaces(data: npt.NDArray):
    import re

    n_data = data.shape[0]
    result = data.copy()
    for i in range(n_data):
        result[i] = re.sub(r"^\s*|\s\s*", " ", result[i])
    return result


# Expanding contractions
def fix_contractions(data: npt.NDArray):
    n_data = data.shape[0]
    result = data.copy()
    for i in range(n_data):
        result[i] = " ".join([contraction_fix(word) for word in result[i].split()])
    return result


def tokenize(data: npt.NDArray):
    tokenized_documents = list()
    for document in data:
        tokenized_documents.append(np.array(document.split()))
    return np.array(tokenized_documents, dtype=object)
