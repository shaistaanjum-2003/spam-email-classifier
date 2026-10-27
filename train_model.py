
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/spam_ham_dataset.csv")

X = df["text"]
y = df["label"]

print("Input data:", X.shape)
print("Target data:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training emails:", len(X_train))
print("Testing emails:", len(X_test))

# Convert email text into numerical features
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Training feature matrix:", X_train_tfidf.shape)
print("Testing feature matrix:", X_test_tfidf.shape)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create and train the model
model = SVC(kernel="linear")
model.fit(X_train_tfidf, y_train)

# Predict labels for unseen test emails
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import os
import joblib
# Create a folder to store the trained model
os.makedirs("models", exist_ok=True)

# Save the model and TF-IDF vectorizer
joblib.dump(model, "models/spam_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")