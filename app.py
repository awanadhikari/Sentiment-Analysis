from flask import Flask, request, render_template
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemma_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemma_tokens)

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    tfidf_vector = vectorizer.transform([preprocessed_text])
    prediction = model.predict(tfidf_vector)
    return "Positive" if prediction == 1 else "Negative"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    sentiment = predict_sentiment(text)
    return render_template('index.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
