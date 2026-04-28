from flask import Flask, request, render_template
import pickle
import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__)

# Load files
model = pickle.load(open('models/model.pkl', 'rb'))
tfidf = pickle.load(open('models/tfidfvectorizer.pkl', 'rb'))
labelencoder = pickle.load(open('models/labelencoder.pkl', 'rb'))

# NLP setup
stemmer = PorterStemmer()
total_stopwords = set(stopwords.words('english'))
negative_words = {"no", "not", "nor"}
final_stopwords = total_stopwords - negative_words

HTMLTAGS = re.compile('<.*?>')
table = str.maketrans('', '', string.punctuation)
remove_digits = str.maketrans('', '', string.digits)
MULTIPLE_WHITESPACE = re.compile(r"\s+")

# Preprocessing function
def TEXT_PREPROCESSOR(feedback):
    feedback = feedback.lower()
    feedback = feedback.replace("n't", " not")
    feedback = HTMLTAGS.sub('', feedback)
    feedback = feedback.translate(table)
    feedback = feedback.translate(remove_digits)
    feedback = MULTIPLE_WHITESPACE.sub(" ", feedback).strip()
    feedback = [word for word in feedback.split() if word not in final_stopwords]
    feedback = ' '.join([stemmer.stem(word) for word in feedback])
    return feedback

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    processed = TEXT_PREPROCESSOR(text)
    vector = tfidf.transform([processed])
    probs = model.predict_proba(vector)[0]
    confidence = round(max(probs) * 100, 2)

    prediction = model.predict(vector)[0]
    result = labelencoder.inverse_transform([prediction])[0]

    return render_template('index.html', prediction=result)
if __name__ == "__main__":
    app.run(debug=True)