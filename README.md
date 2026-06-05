# Support Ticket Classification & Prioritization System

An end-to-end NLP pipeline that automatically classifies consumer complaints by category and assigns priority levels — built to solve a real operational problem faced by support teams at financial institutions.

---

## The Problem

Support teams at financial companies receive thousands of complaints daily. Manual triage — reading each complaint, deciding which team should handle it, and deciding how urgently — is slow, inconsistent, and expensive. A fraud victim and a general account inquiry should not sit in the same queue.

This system automates triage with two models running in parallel:
1. **Category classifier** — routes the ticket to the right team
2. **Priority classifier** — tells that team how urgently to respond

---

## Results

| Task | Model | Accuracy |
|---|---|---|
| Category Classification | Logistic Regression | **85.02%** |
| Priority Prediction | Random Forest + VADER Sentiment | **90.55%** |

---

## Pipeline Architecture

```
Raw complaint text
        │
        ▼
  Text Cleaning
  (lowercase, remove placeholders, stopwords, lemmatization)
        │
        ├─────────────────────────────────┐
        │                                 │
        ▼                                 ▼
Category TF-IDF                   Priority TF-IDF
Vectorizer (10,000 features)      Vectorizer (5,000 features)
        │                                 │
        ▼                             + VADER Sentiment Scores
Logistic Regression                       │
        │                                 ▼
        ▼                         Random Forest
  Category + Confidence               │
                                        ▼
                                Priority + Confidence
        │                               │
        └──────────────┬────────────────┘
                       ▼
        Route to [Category] team | [Priority] urgency
```

---

## Dataset

**Source:** [CFPB Consumer Complaint Database](https://www.kaggle.com/datasets/namigabbasov/consumer-complaint-dataset)  
**Original size:** 2,023,066 real government-recorded complaints  
**Used:** 21,222 rows (balanced across categories and priority levels)

### Categories
- Bank Accounts and Services
- Credit Card Services
- Credit Reporting
- Debt Collection
- Loans

### Priority Levels
- 🔴 **High** — Fraud, identity theft, legal violations, unauthorized access
- 🟡 **Medium** — Billing disputes, payment errors, account issues
- 🟢 **Low** — General inquiries, status updates, clarifications

---

## Key Engineering Decisions

### 1. Dataset Selection
The original Kaggle Customer Support Ticket dataset was abandoned after discovering its descriptions were synthetically generated with random label assignments. A Logistic Regression model trained on it achieved **20.84% accuracy** — equivalent to random guessing across 5 classes. No model can learn when labels are random.

The CFPB dataset was selected because it contains real complaints with human-assigned labels. Accuracy immediately jumped to 85%+ after switching.

**Lesson: Data quality > data quantity.**

### 2. Class Balancing
The CFPB dataset has severe imbalance — Credit Reporting has 1.2M rows vs Bank Accounts' 158K. Unbalanced training causes the model to predict the majority class for most inputs. Fixed by sampling exactly 8,000 per category and 7,000 per priority level.

### 3. TF-IDF Vocabulary Size
Testing showed increasing `max_features` from 5,000 to 10,000 improved category accuracy by +0.26%. GridSearchCV confirmed C=1.0 was already optimal — the bottleneck was feature richness, not model parameters.

### 4. Priority Label Engineering (Two Experiments)
No priority labels existed in the dataset. Two approaches were compared:

| Experiment | Features | Labels | Best Model | Accuracy |
|---|---|---|---|---|
| A | TF-IDF only | Keyword-based | Random Forest | 89.23% |
| B | TF-IDF + VADER sentiment | Keyword + Sentiment | Random Forest | **90.55%** |

Adding VADER sentiment scores as additional features improved accuracy by **+1.32%**. Sentiment captures emotional intensity that keywords alone miss — "someone has COMPLETELY drained my account" is more urgent than "there may be an unauthorized charge".

### 5. Why Random Forest Won for Priority (But Not Category)
Priority labels were engineered using keyword counting rules. Random Forest's decision tree logic naturally mirrors this rule-based pattern — it learned to replicate our keyword detection logic. Logistic Regression and LinearSVC find linear boundaries which are better suited to semantic patterns but less suited to rule-mimicry.

### 6. Confidence Thresholding
Predictions below 60% confidence are flagged for human review rather than automatically routed. In the demo, 4/8 tickets were auto-routed and 4/8 were flagged. This is a critical production pattern — ML systems should know when they don't know.

---

## Project Structure

```
support-ticket-classifier/
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb       # Data loading, cleaning, EDA
│   ├── 02_category_classification.ipynb # Category model training & evaluation
│   ├── 03_priority_prediction.ipynb     # Priority model with sentiment experiments
│   └── 04_inference_demo.ipynb          # End-to-end inference pipeline
│
├── data/
│   ├── complaints.csv                   # Raw CFPB dataset (download from Kaggle)
│   ├── cleaned_tickets.csv              # Generated by Notebook 1
│   ├── distribution_plot.png
│   ├── wordclouds_cleaned.png
│   ├── category_model_comparison.png
│   ├── category_confusion_matrix.png
│   ├── priority_experiment_comparison.png
│   └── priority_confusion_matrix.png
│
├── models/
│   ├── category_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── category_label_encoder.pkl
│   ├── priority_model.pkl
│   ├── priority_tfidf_vectorizer.pkl
│   ├── priority_label_encoder.pkl
│   └── priority_config.pkl
│
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/support-ticket-classifier
cd support-ticket-classifier
```

### 2. Set up environment
```bash
uv venv --python 3.11
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

uv pip install pandas numpy matplotlib seaborn scikit-learn nltk spacy wordcloud jupyter ipykernel vaderSentiment
python -m spacy download en_core_web_sm
```

### 3. Download dataset
Download `complaints.csv` from [Kaggle](https://www.kaggle.com/datasets/namigabbasov/consumer-complaint-dataset) and place it in the `data/` folder.

### 4. Run notebooks in order
```
01_eda_preprocessing.ipynb      → generates cleaned_tickets.csv
02_category_classification.ipynb → generates category models
03_priority_prediction.ipynb    → generates priority models
04_inference_demo.ipynb         → end-to-end demo
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | TF-IDF, model training, evaluation |
| NLTK | Tokenization, stopwords, lemmatization |
| VADER Sentiment | Complaint urgency scoring |
| matplotlib / seaborn | Visualizations |
| WordCloud | EDA visualization |
| joblib | Model serialization |

---

## Business Impact

A support team processing 1,000 tickets per day manually spends 4-6 hours on triage alone. This pipeline automates triage at 85%+ accuracy, reducing triage time to near zero. High priority fraud and identity theft cases are automatically escalated — reducing response time for the most vulnerable customers.

At scale (10,000 tickets/day), even a 1% misclassification rate means 100 tickets per day incorrectly routed. The confidence thresholding feature ensures borderline cases are reviewed by humans rather than silently misrouted.

---

## Author

**Hridhayansh Reddi B**  
2nd Year CSE/SE Student, VIT Chennai  
[LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)
