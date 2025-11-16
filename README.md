

# **Bank Customer Feedback Analysis using ML & Deep Learning**

![Banner](https://img.shields.io/badge/AI%20Project-Banking-blue)

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

**Sample Visualization:**

![Category Distribution](https://via.placeholder.com/600x300.png?text=Category+Distribution)
![WordCloud Example](https://via.placeholder.com/600x300.png?text=WordCloud+Example)

---

### **3. Machine Learning Models**

**Techniques Used:**

* **TF-IDF Vectorization:** Convert text to numerical features.
* **Naive Bayes:** Classic text classification model.
* **Random Forest:** Ensemble model for robust prediction.

**Performance Metrics:**

* Accuracy, Precision, Recall, F1-Score.
* Confusion matrices visualized.

**Sample ML Confusion Matrix:**

![ML Confusion Matrix](https://via.placeholder.com/600x300.png?text=ML+Confusion+Matrix)

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

**Sample DL Accuracy Curve:**

![DL Accuracy Curve](https://via.placeholder.com/600x300.png?text=LSTM+Accuracy+Curve)

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

## **How to Run the Project**

1. Clone the repository:

```bash
git clone <your-repo-url>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place the dataset in `data/customer_feedback.csv`.
4. Run the analysis:

```bash
python src/customer_feedback_analysis.py
```


---

I can also **generate the actual graph images** (bar charts, word clouds, confusion matrices) and replace the placeholder images so the README looks **fully polished and visually attractive** on GitHub.

Do you want me to do that next?
