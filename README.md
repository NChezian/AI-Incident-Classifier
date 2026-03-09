# 🔍 AI-Powered IT Incident Log Classifier

An NLP-based system that automatically classifies IT incident logs by **category**, **priority level**, and **suggested resolution team** — simulating intelligent incident triage for enterprise IT operations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

[App Screenshot](Test_screenshot.png)

## 📋 Business Problem

In enterprise IT environments, helpdesk teams receive hundreds of incident tickets daily. Manual triage is:

- **Slow** — analysts read and categorise each ticket individually
- **Inconsistent** — different analysts may assign different priorities
- **Costly** — misrouted tickets increase mean time to resolution (MTTR)

This project demonstrates how **NLP + ML** can automate the initial triage step, providing instant classification to accelerate incident response workflows.

---

## 🏗️ Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│  Incident     │────▶│  TF-IDF       │────▶│  Logistic        │
│  Description  │     │  Vectoriser   │     │  Regression      │
│  (free text)  │     │  (1,2)-grams  │     │  (multi-class)   │
└──────────────┘     └───────────────┘     └──────┬───────────┘
                                                   │
                                    ┌──────────────┼──────────────┐
                                    ▼              ▼              ▼
                              ┌──────────┐  ┌──────────┐  ┌──────────────┐
                              │ Category │  │ Priority │  │ Suggested    │
                              │ Network  │  │ Critical │  │ Team         │
                              │ Software │  │ High     │  │ (mapped from │
                              │ Hardware │  │ Medium   │  │  category)   │
                              │ Access   │  │ Low      │  └──────────────┘
                              └──────────┘  └──────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/NChezian/ai-incident-classifier.git
cd ai-incident-classifier
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python src/generate_dataset.py
```

Creates `data/it_incidents.csv` with 800 synthetic IT incident records across 4 categories.

### 3. Train Models

```bash
python src/train_model.py
```

Outputs trained models and metrics to `models/`.

### 4. Launch App

```bash
streamlit run src/app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Model Performance

| Classifier | Accuracy | Algorithm |
|:-----------|:---------|:----------|
| Category   | 100%     | TF-IDF + Logistic Regression |
| Priority   | 100%     | TF-IDF + Logistic Regression |

> **Note:** High accuracy reflects the structured synthetic dataset. Real-world performance would be lower due to noisy, ambiguous ticket text. In production, the model would be retrained on actual helpdesk data, and priority classification would incorporate additional signals (affected user count, SLA tier, asset criticality).

---

## 📁 Project Structure

```
ai-incident-classifier/
├── data/
│   └── it_incidents.csv          # Generated training data
├── models/
│   ├── category_model.joblib     # Trained category classifier
│   ├── priority_model.joblib     # Trained priority classifier
│   ├── team_map.json             # Category → Team mapping
│   └── metrics.json              # Model evaluation metrics
├── src/
│   ├── generate_dataset.py       # Synthetic data generator
│   ├── train_model.py            # Model training pipeline
│   └── app.py                    # Streamlit web application
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔮 Potential Extensions

- **Real dataset integration** — Connect to ServiceNow / Jira APIs for live ticket data
- **Deep learning** — Replace Logistic Regression with a fine-tuned transformer (BERT/DistilBERT)
- **Feedback loop** — Let analysts correct predictions to retrain the model incrementally
- **Deployment** — Containerise with Docker, deploy on AWS/Azure as a microservice
- **Explainability** — Add LIME/SHAP to highlight which words drove the classification

---

## 🛠️ Tech Stack

| Component         | Technology              |
|:------------------|:------------------------|
| Language          | Python 3.10+            |
| NLP               | TF-IDF (scikit-learn)   |
| ML Model          | Logistic Regression     |
| Data Processing   | Pandas, NumPy           |
| Web UI            | Streamlit               |
| Model Persistence | Joblib                  |

---

## 👤 Author

**Nikhil Chezian**
- 📍 Heidelberg, Germany
- 🎓 M.Eng. Information Technology — SRH Hochschule Heidelberg
- 💼 Background in ML Engineering, IT Operations & System Delivery
- 📧 chezian.nikhil@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/nikhil-chezian/) | [GitHub](https://github.com/NChezian)

---

## 📄 License

This project is licensed under the MIT License.
