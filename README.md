Hyperglycaemia Risk Prediction 🩺

""Python" (https://img.shields.io/badge/python-3.9%2B-blue?logo=python)"
""License" (https://img.shields.io/badge/license-MIT-green)"

«Predictive system to forecast hyperglycaemia risk using classical ML models + LSTM, with a web UI and automated reporting.»

📋 Table of Contents

1. "Project Overview" (#project-overview)
2. "Motivation & Objective" (#motivation--objective)
3. "Data & Preprocessing" (#data--preprocessing)
4. "Models & Techniques" (#models--techniques)
5. "Class Imbalance Handling" (#class-imbalance-handling)
6. "Results & Evaluation" (#results--evaluation)
7. "Demo / UI Screenshot" (#demo--ui-screenshot)
8. "Installation & Usage" (#installation--usage)
9. "Project Structure" (#project-structure)
10. "Future Improvements" (#future-improvements)
11. "License & Contact" (#license--contact)

---

📈 Project Overview

This repository contains an end-to-end machine learning solution developed as part of my MSc Data Science dissertation.
It predicts hyperglycaemia risk from time-series diabetic patient data — using multiple classical ML algorithms and a deep-learning LSTM model — and delivers results through an interactive Streamlit web interface. Users can input patient data, receive risk predictions, and download a PDF clinical-style report summarising the results.

This pipeline covers:
Data collection & processing → model development & benchmarking → application deployment → reporting

---

🔬 Motivation & Objective

- Address a real-world health challenge: early detection of hyperglycaemia risk.
- Compare classical ML models against deep learning for time-series medical data.
- Build a usable tool (not just research) — actionable risk predictions + reporting.
- Demonstrate full-cycle data science: from raw data to product-level deployment.

---

🧰 Data & Preprocessing

- Time-series dataset (anonymized / synthetic-based for privacy) with multiple patient observations across time.
- Feature engineering: rolling windows, temporal features, normalization/scaling.
- Data cleaning, missing value handling, and date/time parsing.
- Train/test split with stratification by class label and time to avoid data leakage.

---

🧠 Models & Techniques

Trained, tuned, and evaluated 10+ models:

#| Model| Type
1| Logistic Regression (multinomial)| Baseline ML
2| Random Forest| Ensemble
3| Gradient Boosting| Ensemble
4| Extra Trees Classifier| Ensemble
5| AdaBoost Classifier| Ensemble
6| Support Vector Machine (SVM, RBF)| ML
7| K-Nearest Neighbours (KNN)| ML
8| Naive Bayes| ML
9| XGBoost| Gradient Boosting
10| LSTM (Deep Learning)| Time-series DL

Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC where applicable.

---

⚖️ Class Imbalance Handling

The dataset had an imbalanced distribution between normal and high-risk classes. To address this, the following techniques were used:

- SMOTE (Synthetic Minority Over-sampling Technique)
- RandomOverSampler (fallback when SMOTE constraints apply)

Result: Balanced target distribution, improved recall and better generalisation on minority class (high-risk) — critical for healthcare applications.

---

📊 Results & Evaluation (Summary)

- LSTM model achieved highest overall performance (best recall + F1-score on high-risk class).
- Tree-based models (Random Forest, XGBoost) were the strongest classical models — good baseline alternatives.
- After class balancing, recall for high-risk class improved by ~20–25% over baseline.
- Mixed-model comparison results, confusion matrices, ROC & PR curves included for transparency.

(See "confusion_matrices_*.png", "all_models_metrics_comparison.png" for details)

---

🎯 Demo / UI Screenshot

"Streamlit UI screenshot" (path/to/your_screenshot.png)  <!-- Add your actual image path -->

Enter patient parameters → Get real-time risk prediction → Download PDF report

---

🚀 Installation & Usage

git clone https://github.com/Chethancj/Hyperglycemia-risk-prediction-using-logistic-regression-and-a-prediction-calculator.git
cd Hyperglycemia-risk-prediction-using-logistic-regression-and-a-prediction-calculator
pip install -r requirements.txt
streamlit run app.py  # or streamlit run app/streamlit_app.py depending on file structure

---

📂 Project Structure

/  
  ├── data/                  # Data files  
  ├── notebooks/             # EDA & prototyping  
  ├── models/                # Saved model artifacts (.pkl, .h5)  
  ├── src/                   # Scripts: preprocessing, training, evaluation, inference  
  ├── app.py                 # Streamlit application for risk prediction  
  ├── requirements.txt       # Dependencies  
  └── README.md              # Project documentation  

---

🔮 Future Improvements

- Add cross-validation and hyperparameter tuning pipelines (e.g. grid/random search)
- Extend to more advanced deep-learning architectures (GRU, Transformer-based time-series)
- Add automated testing + CI/CD for model training and deployment
- Expand dataset to include more patient data, validation on external cohorts
- Add user authentication and secure data storage for a production-grade health application

---

📄 License & Contact

This project is released under the MIT License — feel free to reuse, modify or distribute.

GitHub: https://github.com/Chethancj
LinkedIn: https://linkedin.com/in/chethan-jayashankar

--
If you found this useful — ⭐ Star the repo and let’s connect!
