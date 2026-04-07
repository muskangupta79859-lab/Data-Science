"""
Simple Sentiment Analysis Model
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():
    # Sample dataset
    texts = [
        "I love this product",
        "This is amazing",
        "I hate this",
        "Very bad experience",
        "I really like it",
        "Worst product ever"
    ]

    labels = [1, 1, 0, 0, 1, 0]  # 1 = positive, 0 = negative

    # Convert text to numbers
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test prediction
    test_text = ["I love it"]
    test_vec = vectorizer.transform(test_text)

    prediction = model.predict(test_vec)

    print("Prediction:", "Positive" if prediction[0] == 1 else "Negative")


if __name__ == "__main__":
    main()
