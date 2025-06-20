# -*- coding: utf-8 -*-
"""DeepLearning_Lena-Aucéane.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tQ8tfKtghN8xQ19TzGA9nxPS3L2Fgxdv
"""

#!pip install --upgrade pip
#!pip install pandas numpy matplotlib seaborn wordcloud
#!pip install nltk
#!pip install tensorflow scikit-learn
#!pip install transformers datasets
#!pip install streamlit
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# %% --------------Traitement des données et Analyse De Sentiments
import matplotlib.pyplot as plt
import seaborn as sns

finance_data = pd.read_csv("C:/Users/Lena/Desktop/deeplearning/stock_yfinance_data.csv")
tweets_data = pd.read_csv("C:/Users/Lena/Desktop/deeplearning/stock_tweets.csv")

# Vérification du chargement
print("Aperçu des tweets :")
print(tweets_data.head())

print("\nAperçu des données financières :")
print(finance_data.head())

# Informations générales sur le dataset
tweets_data.info()

# Dimensions du dataset (lignes, colonnes)
print("\nDimensions du dataset :", tweets_data.shape)

# Aperçu des 10 premières lignes
tweets_data.head(10)

# Vérification des valeurs manquantes
tweets_data.isnull().sum()

finance_data.head()

#!pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# On initialise l'analyseur de sentiment VADER
analyzer = SentimentIntensityAnalyzer()

# Fonction qui retourne à la fois le score et le label
def compute_sentiment(text):
    score = analyzer.polarity_scores(str(text))['compound']
    if score >= 0.05:
        label = 'Positive'
    elif score <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
    return score, label

# On applique la fonction
tweets_data[['Sentiment_Score', 'Sentiment']] = tweets_data['Tweet'].apply(
    lambda x: pd.Series(compute_sentiment(x))
)

# Vérifions les nouvelles colonnes
tweets_data.head()

# Comptage des labels de sentiment
tweets_data['Sentiment'].value_counts()
tweets_data['Sentiment'].value_counts(normalize=True) * 100

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(x='Sentiment', data=tweets_data, order=['Positive', 'Neutral', 'Negative'])
plt.title("Répartition des tweets par sentiment")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(tweets_data['Sentiment_Score'], bins=30, kde=True)
plt.title("Distribution des scores de sentiment (VADER)")
plt.show()

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Liste des stopwords anglais
stop_words = set(stopwords.words('english'))

# Fonction de nettoyage
def clean_text(text):
    text = text.lower()  # Minuscule
    text = re.sub(r'http\S+', '', text)  # Supprimer les URLs
    text = re.sub(r'@\w+', '', text)  # Supprimer les mentions
    text = re.sub(r'#', '', text)  # Supprimer le # mais garder le mot
    text = re.sub(r'\d+', '', text)  # Supprimer les chiffres
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Supprimer les espaces multiples
    # Supprimer les stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Création d'une nouvelle colonne Clean_Tweet
tweets_data['Clean_Tweet'] = tweets_data['Tweet'].apply(clean_text)

# Vérifions
tweets_data[['Tweet', 'Clean_Tweet']].head(10)

# ---------------------
# 3. Prétraitement des dates 
# ---------------------

tweets_data['Date'] = pd.to_datetime(tweets_data['Date']).dt.tz_localize(None)
finance_data['Date'] = pd.to_datetime(finance_data['Date']).dt.tz_localize(None)


# ---------------------
# 5. Fusion des datasets
# ---------------------

merged = pd.merge(tweets_data, finance_data, on='Date')

merged.to_csv("C:/Users/Lena/Desktop/deeplearning/merged.csv", index=False)

# %% --------------DEEP lEARNING: Keras prediction hausse/baisse

# ---------------------
# 6. Création de la target Hausse/Baisse
# ---------------------

merged = merged.sort_values(by='Date')
merged['Next_Close'] = merged['Close'].shift(-1)
merged['Target'] = (merged['Next_Close'] > merged['Close']).astype(int)
merged.dropna(inplace=True)

# ---------------------
# 7. Sélection des features (que des valeurs numériques !)
# ---------------------

features = ['Sentiment_Score', 'Open', 'High', 'Low', 'Close', 'Volume']
X = merged[features]
y = merged['Target']

# ---------------------
# 8. Standardisation des données (indispensable en Deep Learning)
# ---------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------
# 9. Découpage Train/Test
# ---------------------

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------
# 10. Construction du modèle Deep Learning
# ---------------------

nb_features = X_train.shape[1]

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(nb_features,)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # binaire

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ---------------------
# 11. Entraînement
# ---------------------

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# ---------------------
# 12. Évaluation du modèle
# ---------------------

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Sauvegarde du modèle
model.save("hausse_baisse_model.keras")

import joblib

# Sauvegarde du scaler
joblib.dump(scaler, "hausse_baisse_scaler.pkl")

#Test Accuracy: 72.50%


# %% --------------DEEP lEARNING: BERT prediction hausse/baisse Test Accuracy: 76.92% 

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Chargement du tokenizer et du modèle FinBERT
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")


merged['Clean_Tweet'] = merged['Tweet'].apply(clean_text)
merged['Clean_Tweet'] = merged['Tweet'].apply(clean_text)

# --------------------
# 3 - Réduction de l’échantillon pour test FinBERT
# --------------------
sampled_data = merged.sample(n=390, random_state=42).copy()

# --------------------
# 4 - Application de FinBERT
# --------------------
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

def get_finbert_sentiment(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = finbert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    score = probs[0][2].item() - probs[0][0].item()  # Positive - Negative
    return score

sampled_data['FinBERT_Score'] = sampled_data['Tweet'].apply(get_finbert_sentiment)

# --------------------
# 5 - Création de la target Hausse/Baisse
# --------------------
sampled_data = sampled_data.sort_values(by='Date')
sampled_data['Next_Close'] = sampled_data['Close'].shift(-1)
sampled_data['Target'] = (sampled_data['Next_Close'] > sampled_data['Close']).astype(int)
sampled_data.dropna(inplace=True)

# --------------------
# 6 - Préparation du dataset final
# --------------------
features = ['FinBERT_Score', 'Open', 'High', 'Low', 'Close', 'Volume']
X = sampled_data[features]
y = sampled_data['Target']

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --------------------
# 7 - Deep Learning modèle
# --------------------
nb_features = X_train.shape[1]

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(nb_features,)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")