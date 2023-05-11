import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import csv
import os
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix


# Get current working directory
cwd = os.getcwd()
print("Current working directory: %s \n" % cwd)


def collect_and_preprocess_data():
    reviews_df = pd.read_csv(cwd + '/reviews.csv', delimiter='\t')
    reviews_df = reviews_df.sort_values(by=['RatingValue'])
    reviews_df['Sentiment'] = pd.cut(
        reviews_df['RatingValue'], bins=[
            0, 2, 3, 5], labels=[
            0, 1, 2])

    positive_ratings = len(reviews_df[reviews_df.Sentiment == 2])
    neutral_ratings = len(reviews_df[reviews_df.Sentiment == 1])
    negative_ratings = len(reviews_df[reviews_df.Sentiment == 0])
    num_ratings = [positive_ratings, neutral_ratings, negative_ratings]

    positive_rows_to_drop = reviews_df[reviews_df['Sentiment']
                                       == 2].index[:positive_ratings - min(num_ratings)]
    nuetral_rows_to_drop = reviews_df[reviews_df['Sentiment']
                                      == 1].index[:neutral_ratings - min(num_ratings)]
    negative_rows_to_drop = reviews_df[reviews_df['Sentiment']
                                       == 0].index[:negative_ratings - min(num_ratings)]

    reviews_df = reviews_df.drop(positive_rows_to_drop)
    reviews_df = reviews_df.drop(nuetral_rows_to_drop)
    reviews_df = reviews_df.drop(negative_rows_to_drop)

    positive_ratings = len(reviews_df[reviews_df.Sentiment == 2])
    neutral_ratings = len(reviews_df[reviews_df.Sentiment == 1])
    negative_ratings = len(reviews_df[reviews_df.Sentiment == 0])
    num_ratings = [positive_ratings, neutral_ratings, negative_ratings]
    balanced_reviews = reviews_df[["Sentiment", "Review"]]
    return balanced_reviews


# Call collect and prerpocess data
balanced_reviews = collect_and_preprocess_data()
train, valid = train_test_split(balanced_reviews, test_size=0.2)

# Create test.csv and train.csv
train.to_csv("train.csv", index=False)
valid.to_csv("valid.csv", index=False)

# Write test.csv and train.csv
train = pd.read_csv("train.csv")
valid = pd.read_csv("valid.csv")

# Define X_train, y_train, X_valid, y_valid
X_train = train['Review']
y_train = train['Sentiment']
X_valid = valid['Review']
y_valid = valid['Sentiment']

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train.Review)
tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, train.Sentiment)
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
text_clf.fit(train.Review, train.Sentiment)
docs_test = valid.Review
predicted = text_clf.predict(docs_test)

# Print all required metrics
print("Accuracy: " + str(np.mean(predicted == valid.Sentiment)))
print("Macro F1 Score: " + str(f1_score(predicted, valid.Sentiment, average='macro')))
target_names = ['Negative', 'Neutral', 'Positive']
print("Classwise F1 Score In Full Classification Report Below:\n")
print(
    classification_report(
        valid.Sentiment,
        predicted,
        target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(valid.Sentiment, predicted, normalize='true'))
