import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
from spacy.language import Language
from flask import Flask, render_template, request
import langid

# Load WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the Keras model
model = load_model('model.h5')

# Load intents, words, and classes
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)
app.static_folder = 'static'

# Load English to Swahili translation pipeline
eng_swa_model_checkpoint = "Helsinki-NLP/opus-mt-en-swc"
eng_swa_tokenizer = AutoTokenizer.from_pretrained(eng_swa_model_checkpoint)
eng_swa_model = AutoModelForSeq2SeqLM.from_pretrained(eng_swa_model_checkpoint)
eng_swa_translator = pipeline("text2text-generation", model=eng_swa_model, tokenizer=eng_swa_tokenizer)

# Load Swahili to English translation pipeline
swa_eng_model_checkpoint = "Helsinki-NLP/opus-mt-swc-en"
swa_eng_tokenizer = AutoTokenizer.from_pretrained(swa_eng_model_checkpoint)
swa_eng_model = AutoModelForSeq2SeqLM.from_pretrained(swa_eng_model_checkpoint)
swa_eng_translator = pipeline("text2text-generation", model=swa_eng_model, tokenizer=swa_eng_tokenizer)

def clean_up_sentence(sentence):
    return [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)]

def bow(sentence, words):
    bag = [0] * len(words)
    for i, w in enumerate(words):
        if w in sentence:
            bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    return [{'intent': classes[i], 'probability': str(res[i])} for i, r in enumerate(res) if r > ERROR_THRESHOLD]

def getResponse(ints, intents_json):
    return random.choice([i['responses'] for i in intents_json['intents'] if i['tag'] == ints[0]['intent']])

def chatbot_response(msg):
    detected_language, confidence = langid.classify(msg)
    print(f"Detected language: {detected_language}, Confidence: {confidence}")

    if confidence < 0.5 or detected_language not in ['en', 'sw']:  # Fallback to English if confidence is low or language is unknown
        res = getResponse(predict_class(clean_up_sentence(msg), model), intents)
    elif detected_language == "en":
        res = getResponse(predict_class(clean_up_sentence(msg), model), intents)
    elif detected_language == 'sw':
        translated_msg = swa_eng_translator(msg, max_length=128, num_beams=5)[0]['generated_text']
        res = eng_swa_translator(getResponse(predict_class(clean_up_sentence(translated_msg), model), intents), max_length=128, num_beams=5)[0]['generated_text']

    return res

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print("User input:", userText)

    bot_response = chatbot_response(userText)

    if bot_response is None:
        bot_response = "I'm sorry, I couldn't understand that."

    return bot_response

if __name__ == "__main__":
    app.run()
