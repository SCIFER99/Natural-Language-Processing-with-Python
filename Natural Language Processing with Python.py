# By: Tim Tarver
# Natural Language Processing with Python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create the Dataset

dataset = [("I love this cartoon!", "positive"),
           ("This cartoon is terrible!", "negative"),
           ("I have mixed feelings about this cartoon.", "neutral"),
           ("This program is not biased.", "predictions"),
           ("I love You!", "positive"),
           ]

# Split the data into its Training and Testing Splits
texts, labels = zip(*dataset)
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)

# Extract features using TF-IDF
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_texts)
test_vectors = vectorizer.transform(test_texts)

# Train a classifier to classify the text(Support Vector Machine in this case)
classifier = SVC()
classifier.fit(train_vectors, train_labels)

# Predict and Evaluate the Accuracy
predictions = classifier.predict(test_vectors)
accuracy = accuracy_score(test_labels, predictions)
print("Prediction:", predictions)
print("Accuracy:", accuracy)
