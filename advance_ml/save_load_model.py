"""
Save and Load ML Model
"""

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_and_save():
    texts = [
        "I love this product",
        "This is amazing",
        "I hate this",
        "Very bad experience",
        "I really like it",
        "Worst product ever"
    ]

    labels = [1, 1, 0, 0, 1, 0]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save vectorizer
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)


def load_and_predict():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    text = ["I really love this"]
    text_vec = vectorizer.transform(text)

    prediction = model.predict(text_vec)

    print("Prediction:", "Positive" if prediction[0] == 1 else "Negative")


if __name__ == "__main__":
    train_and_save()
    load_and_predict()
