Hyperglycaemia Risk Prediction Using Machine Learning & LSTM

This project focuses on predicting hyperglycaemia risk using real-world, time-series patient data and a comprehensive range of machine learning and deep learning models.

It was developed as part of my MSc Data Science dissertation and demonstrates a full, production-style ML pipeline including class imbalance handling, model benchmarking, and deployment via Streamlit with automated clinical report generation.

---

🚀 Project Objectives

- Predict hyperglycaemia risk from time-series patient data
- Handle real-world class imbalance using resampling techniques
- Compare multiple machine learning models with deep learning (LSTM)
- Provide an interactive, real-time prediction system
- Generate automated PDF-based clinical reports
- Build a complete pipeline:
Data → Preprocessing → Modelling → Application → Report

---

🧠 Models Implemented (8+ Models)

The following models were trained, tuned, and benchmarked:

1. Logistic Regression (multinomial, saga solver)
2. Random Forest
3. Gradient Boosting
4. Extra Trees Classifier
5. AdaBoost Classifier
6. Support Vector Machine (SVM – RBF kernel)
7. K-Nearest Neighbours (KNN)
8. Naive Bayes
9. XGBoost
10. LSTM (Deep Learning for time-series data)

All models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC (where applicable)

---

⚖️ Handling Class Imbalance

To address class imbalance in the dataset (a common real-world challenge), the following techniques were implemented:

- SMOTE (Synthetic Minority Over-sampling Technique)
- RandomOverSampler (fallback when SMOTE not applicable)

This ensured:

- Balanced class distribution
- Improved model generalisation
- Reduced bias towards majority class
- Better detection of high-risk cases

Code includes:

SMOTE(random_state=42)
RandomOverSampler(random_state=42)

This significantly improved recall for high-risk hyperglycaemia cases.

---

⚙️ Technologies Used

- Python
- TensorFlow / Keras (LSTM)
- scikit-learn
- XGBoost
- Pandas, NumPy
- SMOTE, RandomOverSampler (imbalanced-learn)
- Streamlit (Web interface)
- Matplotlib / Seaborn (Visualisation)
- FPDF / ReportLab (PDF report generation)

---

📊 System Features

✅ Time-series preprocessing & scaling
✅ Class imbalance handling using SMOTE
✅ Multi-model training & benchmarking
✅ Hyperglycaemia risk classification
✅ Interactive Streamlit web interface
✅ Automated, downloadable PDF report
✅ Modular, reusable and reproducible pipeline

---

▶️ How to Run

git clone https://github.com/Chethancj/hyperglycemia-streamlit-user-interface.git
cd hyperglycemia-streamlit-user-interface
pip install -r requirements.txt
streamlit run app/streamlit_app.py

---

📈 Results Summary (example – edit if you like)

- LSTM achieved the highest overall performance
- Tree-based models (Random Forest, XGBoost) were the top classical performers
- Balanced dataset improved recall in high-risk class by ~20% after applying SMOTE
- Demonstrated robust generalisation on unseen data

---

🎯 Real-World Relevance

This project demonstrates advanced skills in:

- Healthcare machine learning
- Time-series modelling
- Imbalanced data handling
- End-to-end ML system development
- Clinical decision support
- Applied AI for early risk detection

Applicable to:

- HealthTech
- AI/ML Engineering
- Research roles
- Predictive analytics teams

---

👤 Author

Chethan Jayashankar
Data Scientist | Machine Learning | AI & LLM Solutions
United Kingdom

LinkedIn: https://linkedin.com/in/cheathan-jayashankar

---

⭐ If you find this project useful, feel free to star the repository and connect with me on LinkedIn.
