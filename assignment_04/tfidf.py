from typing import List, Dict, Set

import numpy as np
import numpy.typing as npt


class TfIdf:
    # Implement low frequency terms and other techniques
    def __init__(self) -> None:
        self.n_docs: int = None
        self.vocab: List = list()
        self.vocab_size: int = None
        self.vocab_index: Dict[str, int] = dict()
        self.word_document_count: Dict[str, int] = dict()

    def __create_vocab__(self, documents: npt.NDArray) -> Set:
        vocab = set()

        for document in documents:
            for word in document:
                vocab.add(word)

        return list(vocab)

    def __get_word_document_count__(self, documents: npt.NDArray):
        word_document_count = dict()

        for document in documents:
            for word in document:
                if word in self.vocab:
                    if word not in word_document_count:
                        word_document_count[word] = 1
                    else:
                        word_document_count[word] += 1

        return word_document_count

    def __term_frequency__(self, word: str, document: npt.NDArray):
        word_occurences = (document == word).sum()
        return word_occurences / self.n_docs

    def __inverse_document_frequency__(self, word: str):
        word_occurrences = 1

        if word in self.word_document_count:
            word_occurrences += self.word_document_count[word]

        return np.log(self.n_docs / word_occurrences)

    def __tf_idf__(self, document: npt.NDArray):
        tf_idf_vector = np.zeros(shape=(self.vocab_size,))
        for word in document:
            # ignore word not in vocab
            if word in self.vocab:
                tf = self.__term_frequency__(word, document)
                idf = self.__inverse_document_frequency__(word)

                tf_idf_vector[self.vocab_index[word]] = tf * idf
        return tf_idf_vector

    def fit(self, documents: npt.NDArray):
        self.n_docs = documents.shape[0]
        self.vocab = self.__create_vocab__(documents)
        self.vocab_size = len(self.vocab)
        self.vocab_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.word_document_count = self.__get_word_document_count__(documents)

    def transform(self, documents: npt.NDArray):
        tf_idf_vectors = list()
        for document in documents:
            tf_idf_vectors.append(self.__tf_idf__(document))
        return np.array(tf_idf_vectors)

    def export(self):
        return {
            "n_docs": self.n_docs,
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "vocab_index": self.vocab_index,
            "word_document_count": self.word_document_count,
        }

    def load(self, tf_idf_model_data):
        self.n_docs = tf_idf_model_data["n_docs"]
        self.vocab_size = tf_idf_model_data["vocab_size"]
        self.vocab = tf_idf_model_data["vocab"]
        self.vocab_size = tf_idf_model_data["vocab_size"]
        self.vocab_index = tf_idf_model_data["vocab_index"]
        self.word_document_count = tf_idf_model_data["word_document_count"]
