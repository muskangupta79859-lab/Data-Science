"""
Basic NLP Text Preprocessing
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# First time run
nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    return filtered_words


def main():
    text = "This is a simple example to demonstrate NLP preprocessing."

    processed = preprocess_text(text)

    print("Original Text:\n", text)
    print("\nProcessed Text:\n", processed)


if __name__ == "__main__":
    main()
