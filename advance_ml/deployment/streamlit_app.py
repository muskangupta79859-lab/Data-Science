import streamlit as st
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("Sentiment Analysis App")

# User input
user_input = st.text_input("Enter a sentence:")

if st.button("Predict"):
    if user_input:
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)[0]

        result = "Positive" if prediction == 1 else "Negative"
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text")
