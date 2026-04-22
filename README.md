# Medical Diagnosis System
**FAST-NUCES Karachi — Artificial Intelligence, Section 6B**

| Member | ID |
|---|---|
| Syed Suhaib Raza | 23k-0621 |
| Mohid Raheel Khan | 23k-3000 |
| Rizwan Vadsariya | 23k-0005 |

---

## About
A **Bayesian Network (Naive Bayes)** based medical diagnosis system that predicts the most likely disease from user-selected symptoms, along with confidence probabilities for the top 10 diseases.

- **Dataset:** 4,920 samples · 132 symptoms · 41 diseases
- **Algorithm:** Bernoulli Naive Bayes (Bayesian Network)
- **GUI:** Tkinter (dark-themed)
- **Visualization:** Matplotlib horizontal bar chart

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python main.py
```

> Make sure `Disease_symptom_dataset.csv` is in the **same folder** as `main.py`.

---

## How It Works

### Algorithm: Naive Bayes (Bayesian Network)
The system uses **Bernoulli Naive Bayes**, a probabilistic classifier based on Bayes' theorem:

```
P(Disease | Symptoms) ∝ P(Disease) × ∏ P(Symptomᵢ | Disease)
```

- Each symptom is treated as a binary node (present/absent)
- The disease node is the target variable
- The model learns prior and conditional probabilities from the dataset
- At inference time, it computes `P(Disease | selected symptoms)` for all 41 diseases and returns the ranked probabilities

### System Flow
```
User selects symptoms
        ↓
Binary input vector (132 features)
        ↓
Naive Bayes inference
        ↓
P(Disease | Symptoms) for all 41 diseases
        ↓
Top prediction + confidence + bar chart
```

---

## Features
- 🔍 **Search** symptoms by name
- ✅ **Checkbox** symptom selection
- 📊 **Bar chart** showing top-10 disease probabilities
- 🏥 **Diagnosis card** with disease name and confidence %
- 🗑️ **Clear** button to reset everything
