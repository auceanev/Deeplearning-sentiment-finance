# Deep Learning pour l’Analyse de Sentiment Financier

Ce projet a pour objectif de prédire automatiquement le **sentiment** (positif, neutre, négatif) de **tweets liés à la finance**.  
Nous utilisons à la fois des modèles traditionnels et des approches de **Transfert Learning** (NLP) modernes.

---

## Jeu de données

Dataset utilisé : [Stock Tweets for Sentiment Analysis and Prediction](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction)

- `stock_tweets.csv` : tweets annotés (date, action, score de sentiment)
- `stock_yfinance_data.csv` : données financières (Open, Close, Volume…)

---

## Pipeline

1. **Prétraitement** :
   - Nettoyage texte (URL, mentions, hashtags, stopwords)
   - Tokenisation : Keras / Hugging Face
   - Encodage avec VADER (baseline)

2. **Modélisation** :
   - 🔹 LSTM personnalisé (Embedding + LSTM + Dense)
   - 🔹 DistilBERT fine-tuné (Hugging Face `Trainer`)
   - 🔹 FinBERT (spécifique finance) utilisé pour prédiction de hausse/baisse

3. **Comparaison** :
| Modèle           | Tâche                  | Accuracy | F1-score |
|------------------|------------------------|----------|----------|
| LSTM             | Sentiment (3 classes)  | 0.78     | 0.77     |
| DistilBERT       | Sentiment (3 classes)  | ~        | ~        |
| FinBERT          | Hausse/Baisse (binaire)| 0.769    | –        |

4. **Déploiement Streamlit** :
   - Interface simple avec choix de modèle (LSTM ou DistilBERT)
   - Prédiction en temps réel à partir d’un tweet saisi

---

## 🚀 Lancement de l'application

1. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Lancez l’application :

```bash
streamlit run app.py
```

## 📅 Auteurs

- **Auceane TITOT**
- **Lena DEMANOU**
