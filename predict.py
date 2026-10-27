

import joblib

model = joblib.load("models/spam_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

print("Spam Email Classifier")
print("Type 'exit' to quit.")

while True:
    email = input("\nEnter email text: ")

    if email.strip().lower() == "exit":
        print("Exiting classifier.")
        break

    if not email.strip():
        print("Please enter some email text.")
        continue

    email_tfidf = tfidf.transform([email])
    prediction = model.predict(email_tfidf)[0]

    print("Prediction:", prediction.upper())
