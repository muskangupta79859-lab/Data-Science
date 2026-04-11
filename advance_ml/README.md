# Sentiment Analysis ML Project

## 📌 Overview
This project is a complete end-to-end machine learning application that predicts sentiment (Positive/Negative) from text input.

## 🚀 Features
- Text preprocessing using NLP techniques
- TF-IDF vectorization
- Logistic Regression model
- Model saving and loading
- Flask API for deployment
- Streamlit UI for user interaction

## 🛠️ Technologies Used
- Python
- Scikit-learn
- NLTK
- Flask
- Streamlit

## ▶️ How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run Flask API
python app.py

### 3. Run Streamlit App
streamlit run streamlit_app.py

## 📊 Example
Input: "I love this product"  
Output: Positive

## 📁 Project Structure
advance_ml/
   ├── sentiment_model.py
   ├── save_load_model.py
   ├── tfidf_vectorizer.py
   ├── text_preprocessing.py

deployment/
   ├── app.py
   ├── streamlit_app.py
   ├── model.pkl
   ├── vectorizer.pkl
