# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 13:56:59 2025

@author: Lena
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# -------------------------
# Chargement du modèle LSTM
# -------------------------

# Charger le modèle LSTM (fichier keras)
lstm_model = tf.keras.models.load_model('C:/Users/Lena/Desktop/deeplearning/my_model.keras')

# Charger le tokenizer
with open('tokenizer_lstm.pkl', 'rb') as handle:
    lstm_tokenizer = pickle.load(handle)

# -------------------------
# Fonction de prédiction
# -------------------------

def predict_lstm(text, tokenizer, max_length=50):
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = np.argmax(lstm_model.predict(padded), axis=1)
    return prediction[0]

# -------------------------
# Interface Streamlit
# -------------------------

st.title("📊 Analyse de Sentiment Financier (LSTM uniquement)")

tweet = st.text_area("Entrez le texte du tweet :")

if st.button("Prédire"):
    if tweet.strip() != "":
        pred = predict_lstm(tweet, lstm_tokenizer)
        labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
        st.write("Sentiment prédit :", labels[pred])
    else:
        st.warning("texte à analyser.")
