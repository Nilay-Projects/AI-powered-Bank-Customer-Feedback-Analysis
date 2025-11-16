# src/nlp_analysis_extended.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# For Deep Learning
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

# -----------------------------------
# Load Data
# -----------------------------------
df = pd.read_csv('data/bank_reviews.csv')
print("=== First 5 rows ===")
print(df.head())
print("\n=== Data Info ===")
print(df.info())
print("\n=== Null values ===")
print(df.isnull().sum())

# Create a “category” based on e.g., rating or sentiment if needed
if 'rating' in df.columns:
    def map_rating(r):
        if r >= 4:
            return "Positive"
        elif r <= 2:
            return "Negative"
        else:
            return "Neutral"
    df['category'] = df['rating'].apply(map_rating)
elif 'sentiment' in df.columns:
    df['category'] = df['sentiment']
else:
    # fallback: if there is no rating/sentiment, create a dummy category
    df['category'] = "Unknown"

# Drop rows with missing text or category
df = df.dropna(subset=['review', 'category'])

# Rename for consistency
df = df.rename(columns={'review': 'text'})

# -----------------------------------
# EDA & Visualizations
# -----------------------------------

plt.figure(figsize=(8,6))
sns.countplot(data=df, x='category', order=df['category'].value_counts().index)
plt.title("Feedback Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

print("\nCategory value counts:")
print(df['category'].value_counts())

# WordClouds for each category
for cat in df['category'].unique():
    texts = df[df['category']==cat]['text'].dropna().tolist()
    combined = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud for {cat}")
    plt.show()

# Text length distribution
df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(8,6))
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title("Distribution of Review Length (in words)")
plt.xlabel("Number of words")
plt.ylabel("Frequency")
plt.show()

print("\nText length stats:")
print(df['text_length'].describe())

# -----------------------------------
# Preprocessing & Label Encoding
# -----------------------------------
X = df['text']
y = df['category']

le = LabelEncoder()
y_enc = le.fit_transform(y)
print("\nEncoded label mapping:")
for i, cls in enumerate(le.classes_):
    print(i, cls)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
print("\nTraining set size:", X_train.shape[0], " Test set size:", X_test.shape[0])

# -----------------------------------
# Machine Learning Approach
# -----------------------------------

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Model 1: Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
y_pred_nb = nb.predict(X_test_vec)
print("\n=== Naive Bayes Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb, target_names=le.classes_))

cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(8,6))
sns.heatmap(cm_nb, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_vec, y_train)
y_pred_rf = rf.predict(X_test_vec)
print("\n=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8,6))
sns.heatmap(cm_rf, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap='Oranges')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------------
# Deep Learning Approach (LSTM)
# -----------------------------------
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

model = Sequential([
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    LSTM(32),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("\nDeep Learning model summary:")
model.summary()

history = model.fit(X_train_pad, y_train, epochs=10, batch_size=32,
                    validation_split=0.2, verbose=2)

# Evaluate
loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"\nLSTM Test Accuracy: {acc:.4f}")

y_pred_dl = model.predict(X_test_pad)
y_pred_dl_classes = np.argmax(y_pred_dl, axis=1)
print("\n=== LSTM Classification Report ===")
print(classification_report(y_test, y_pred_dl_classes, target_names=le.classes_))

cm_dl = confusion_matrix(y_test, y_pred_dl_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm_dl, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap='Greens')
plt.title("Confusion Matrix - LSTM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot Loss & Accuracy curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------------
# Save models + vectorizer/tokenizer for later use
import pickle
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('models/nb_model.pkl', 'wb') as f:
    pickle.dump(nb, f)
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
model.save('models/lstm_model.h5')

print("\nModels and vectorizer/tokenizer saved to /models directory.")
