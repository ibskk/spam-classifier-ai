# Spam Detection AI

A machine learning web app that classifies text messages as spam or non-spam using NLP and Naive Bayes.

## Features
- TF-IDF text vectorization
- Naive Bayes classifier
- Train/test split to prevent data leakage
- Confidence-based predictions
- Simple web interface with Streamlit

## Tech Stack
- Python
- scikit-learn
- Pandas
- Streamlit

## How It Works
1. Text messages are converted into numerical vectors using TF-IDF
2. The model learns statistical word patterns from labeled data
3. New messages are classified in real time

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
