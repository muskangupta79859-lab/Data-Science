"""
TF-IDF Vectorization Example
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    documents = [
        "I love machine learning",
        "Machine learning is amazing",
        "I love coding"
    ]

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(documents)

    print("Feature Names:\n", vectorizer.get_feature_names_out())
    print("\nTF-IDF Matrix:\n", X.toarray())


if __name__ == "__main__":
    main()
