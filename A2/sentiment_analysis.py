import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Read in the reviews.csv file
reviews_df = pd.read_csv('reviews.csv')

# Define a function to preprocess the text by removing non-alphanumeric characters and converting to lowercase
def preprocess(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    text = text.lower()
    return text

# Apply the preprocessing function to the review text
reviews_df['Text'] = reviews_df['Text'].apply(preprocess)

# Bin the ratings into negative, neutral, and positive sentiment
reviews_df['Sentiment'] = pd.cut(reviews_df['RatingValue'], bins=[0, 2, 3, 5], labels=[0, 1, 2])

# Split the data into training and testing sets
train_size = int(len(reviews_df) * 0.8)
train_data = reviews_df[:train_size]
test_data = reviews_df[train_size:]

# Train a BoW text classifier on the training set
vectorizer = CountVectorizer()
train_features = vectorizer.fit_transform(train_data['Text'])
classifier = MultinomialNB()
classifier.fit(train_features, train_data['Sentiment'])

# Test the classifier on the testing set
test_features = vectorizer.transform(test_data['Text'])
predictions = classifier.predict(test_features)
accuracy = accuracy_score(test_data['Sentiment'], predictions)
print(f'Test accuracy: {accuracy:.3f}')

# Add the reviews and sentiment labels to a pandas dataframe
reviews_sentiment_df = pd.DataFrame({'Review': reviews_df['Text'], 'Sentiment': reviews_df['Sentiment']})

# Print the first 5 rows of the reviews and sentiment dataframe
print(reviews_sentiment_df.head())
