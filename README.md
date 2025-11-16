

# **Bank Customer Feedback Analysis using ML & Deep Learning**

---

## **Project Overview**

This project analyzes **bank customer feedback** using **Machine Learning (ML)** and **Deep Learning (DL)** techniques. It classifies reviews into categories (e.g., Positive, Negative, Neutral), detects sentiment, and provides actionable insights for banking operations.

**Business Objective:**

* Automatically categorize customer feedback for faster response.
* Identify negative sentiment and potential service issues.
* Enable banks like JPMorgan Chase to improve customer satisfaction and retention.

---

## **Dataset**

* **Source:** [Kaggle – Banks Customer Reviews Dataset](https://www.kaggle.com/datasets/dhavalrupapara/banks-customer-reviews-dataset)
* **Content:** Customer reviews, ratings, and/or sentiment labels.
* **Size:** ~1000+ reviews (can scale with more data).

**Sample Data:**

| text                                 | rating | category |
| ------------------------------------ | ------ | -------- |
| "The credit card process was smooth" | 5      | Positive |
| "Customer service was very slow"     | 1      | Negative |
| "The app is okay, needs updates"     | 3      | Neutral  |

---

## **Project Steps**

### **1. Data Preprocessing**

* Dropped null/missing values.
* Renamed columns for consistency (`review` → `text`).
* Encoded target labels using `LabelEncoder`.
* Mapped ratings/sentiment to categories: Positive / Negative / Neutral.

---

### **2. Exploratory Data Analysis (EDA)**

* **Category Distribution:** Bar chart showing frequency of each feedback type.
* **Word Clouds:** Highlighted common words per category.
* **Text Length Analysis:** Histogram of review word counts.

---

### **3. Machine Learning Models**

**Techniques Used:**

* **TF-IDF Vectorization:** Convert text to numerical features.
* **Naive Bayes:** Classic text classification model.
* **Random Forest:** Ensemble model for robust prediction.

**Performance Metrics:**

* Accuracy, Precision, Recall, F1-Score.
* Confusion matrices visualized.

---

### **4. Deep Learning Model**

**Architecture:**

* **Embedding Layer:** Convert words into vectors.
* **Bi-LSTM:** Capture sequential dependencies.
* **Dense Layers:** For classification.

**Training:**

* Epochs: 10
* Batch Size: 32
* Validation Split: 0.2

**Evaluation:**

* Accuracy and loss curves plotted.
* Confusion matrix for DL predictions.

---

### **5. Model Saving**

* Saved models and vectorizers for future use:

  * `nb_model.pkl` (Naive Bayes)
  * `rf_model.pkl` (Random Forest)
  * `lstm_model.h5` (Deep Learning LSTM)
  * `tfidf_vectorizer.pkl`

---

## **Project Structure**

```
Banking-AI-Project/
│
├── data/
│   └── customer_feedback.csv
├── models/
│   ├── nb_model.pkl
│   ├── rf_model.pkl
│   ├── lstm_model.h5
│   └── tfidf_vectorizer.pkl
├── src/
│   └── customer_feedback_analysis.py
├── notebooks/
│   └── feedback_eda.ipynb
├── requirements.txt
└── README.md
```

---

## **Technologies Used**

* Python
* Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `wordcloud`, `scikit-learn`, `tensorflow`
* ML Models: Naive Bayes, Random Forest
* DL Models: LSTM, Bi-LSTM

---

## **Key Insights**

* Positive reviews are most common, but negative reviews highlight areas for service improvement.
* WordClouds show common terms for each category, e.g., "slow", "delay", "excellent", "helpful".
* DL model slightly outperforms ML models in accuracy and generalization.
* Automated categorization can reduce manual workload and improve response times.

---

## **Uses / Applications**

This project can be used by banks and financial institutions to:

* **Automate Customer Feedback Categorization:** Reduce manual effort in analyzing feedback.
* **Identify Service Gaps:** Quickly detect complaints or negative sentiment.
* **Enhance Customer Experience:** Proactively address pain points and improve satisfaction.
* **Support Decision-Making:** Provide data-driven insights to improve products, apps, and services.
* **Portfolio & Learning:** Showcase NLP, ML, and DL skills for AI, Data Science, and Python roles in banking.

---

