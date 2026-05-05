# ⭐ Text Sentiment Analyzer – NLP & Machine Learning Project

An end-to-end web application that predicts whether a given text is **Positive, Negative, or Neutral** using Natural Language Processing (NLP) and Machine Learning.

---

## 📌 Overview
This project allows users to input text (reviews, feedback, etc.) and get instant sentiment predictions along with a confidence score.  
It demonstrates a complete ML pipeline from **text preprocessing → model building → deployment using Flask**.

---

## 🧠 Machine Learning Pipeline
- Text cleaning (lowercase, remove punctuation, digits, HTML tags)  
- Stopword removal (keeping important words like *not*)  
- Stemming using NLTK  
- Feature extraction using TF-IDF  
- Model training using Logistic Regression  
- Prediction with confidence score  

---

## 🔧 Tech Stack
- Python  
- Flask  
- Scikit-learn (Logistic Regression, TF-IDF)  
- NLTK (stopwords, stemming)  
- HTML, CSS (with basic JavaScript for API integration)  
- Pickle (model saving/loading)  

---

## 🚀 Features
- Real-time sentiment prediction  
- Confidence score for predictions  
- NLP-based text preprocessing and normalization   
- Clean and simple UI  
- End-to-end ML deployment  

---

## ⚙️ Workflow
1. User inputs text  
2. Text preprocessing (cleaning, stopword removal, stemming)  
3. Feature extraction using TF-IDF  
4. Model predicts sentiment  
5. Output displayed with confidence score 

---

## ▶️ Run Locally
```bash
python app.py