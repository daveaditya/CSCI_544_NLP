# Multi-class Classification of Amazon Reviews

Note: Python version used: 3.10.6

The aim of this assignment is to perform multi-class classification of Amazon Review of Jewelry. Important concepts used are of data cleaning, data preprocessing and understanding scoring metrics like precision, recall and f1 score.

The dataset can be found [here](https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Jewelry_v1_00.tsv.gz).

## Data Cleaning

While working with text data in Natural Language Processing, the following techniques are generally used:

- Convert text to lower case
- Remove URLs
- Fix contractions
- Remove non-alphabetical characters
- Remove extra spaces

In addition to this, while going through the reviews, I found that there were certain elements which were not accounted for. They are:

- HTML Encodings (e.g. `&#64;`) - does not add meaning
- HTML Tags especially `<br />` - does not have any semantic meaning
- Emoji Characters (e.g. 👍) - expressions various kinds of emotions
- Accented Characters (e.g. å) - not necessary
- Converting number to NUM tag, e.g. 345 -> NUM

To handle these dataset specific scenarios, I created different functions to handle them. While emoji's were not that frequent and could be removed, a function was developed to convert the emoji to description text e.g. 👍 -> "thumbsup". Accented characters were converted to normal characters to retain information. And HTML related information was discarded.

At the end, we have a clean dataset we could use for further data preprocessing tasks.

## Data Preprocessing

The common data preprocessing tasks are:

- Tokenization
- Removal of Stopwords
- Stemming or Lemmatization

Wordnet tokenizer from NLTK is used for tokenizing. Here I have used `WordNetLemmatizer` for the performing lemmatization. Lemmatization was chosen over stemming as lemmatization gives better results in general. NLTK is used to remove stopwords.

Note: After performing data preprocessing, it was observed that the data sample of 20000 per class was reduced because of empty strings which led to data imbalance. To compensate for that, a total of 21000 reviews per class were selected and then after preprocessing the sampling of 20000 was done.

## Feature Extraction

Here, TF-IDF is used for extraction of features from the documents. It is a good practice to fit the TF-IDF vectorizer on the training set and then use that for transforming the training and test sets. Hence, a 80-20 split was first done, and then features were extacted.

These TF-IDF vectors are then used to train different models.

## Models

The following models were created:

  1. Perceptron
  2. SVM
  3. Logistic Regression
  4. Naive Bayes

While training the models with the default values for hyperparameters, it was observed from the micro averages that all the classes were having different precision values which had good amount of deviation from the average.

Hence, the approach taken was to provide `class weights` for training which would allow to regularize on the classes which were not performing well, and also to penalize those which better, to prevent overfitting. These class weights were then used with Perceptron, SVM, and Logistic Regression and tweaked for each model after calculating the base precision achieved. In addition to that, different values of the regularization constants were also tried.

In the case of `Naive Bayes` different values of `alpha`for adaptive smoothing were tried to see if the average precision of the model could be improved.

In addition to tuning the model, the cleaning and preprocessing pipeline were modified by removing and rearranging the processes to check the possibility of improved performance. A major improvement in performance was seen after removing the step of stopword removal. A speculation here is that the stopwords were somehow dictating the mood of the reviewer, probably by using transitions in the language. I also removed the step of converting numbers to NUM tag as it did not provide any significant gain in precision. While on the otherhand, keeping the process to convert emoji to text proved useful to a certain extend.

## Results

Based on the cleaning -> preprocessing -> model training described above, the Logistic Regression model outperformed other models.

## Conclusion

There are multiple cleaning and preprocessing processes that can be performed on natural language. But it is worth remembering that not all operations are applicable to all the problem / dataset one is working on and it is important to test the ideas and also observe the data for unique traits e.g. existence of HTML tags, or emojis.