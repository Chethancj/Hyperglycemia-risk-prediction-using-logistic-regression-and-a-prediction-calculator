import streamlit as st
import joblib
import numpy as np
import os
import plotly.graph_objects as go

# =========================
# Load Logistic Regression Model
# =========================
MODEL_FILE = "LogisticRegression_Multinomial_model.pkl"  # Change if needed

if not os.path.exists(MODEL_FILE):
    st.error(f"❌ Logistic Regression model file '{MODEL_FILE}' not found. Please upload it.")
else:
    model, scaler = joblib.load(MODEL_FILE)

    # =========================
    # Page Setup
    # =========================
    st.set_page_config(page_title="Hyperglycemia Risk Calculator", page_icon="🩺", layout="wide")
    st.title("🩺 Logistic Regression Hyperglycemia Risk Calculator")
    st.markdown("Enter patient details to get hyperglycemia risk predictions.")

    # =========================
    # Feature Input
    # =========================
    with st.form("risk_form"):
        col1, col2 = st.columns(2)

        with col1:
            glucose = st.number_input("Glucose (mg/dL)", 50, 400, 120)
            carbs = st.number_input("Carbs intake (g)", 0, 300, 50)
            insulin = st.number_input("Insulin dose (units)", 0, 100, 10)
            activity = st.number_input("Activity (mins/day)", 0, 300, 30)
            stress = st.slider("Stress Level (0=Low, 10=High)", 0, 10, 5)
            time_of_day = st.selectbox("Time of Day", [0, 1, 2, 3])
            age = st.number_input("Age", 1, 100, 40)
            bmi = st.number_input("BMI", 10.0, 50.0, 25.0)

        with col2:
            hba1c = st.number_input("HbA1c (%)", 4.0, 15.0, 7.0)
            hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
            kidney_disease = st.selectbox("Kidney Disease (0=No, 1=Yes)", [0, 1])
            type_diabetes = st.selectbox("Type of Diabetes (1=Type1, 2=Type2)", [1, 2])
            duration_since_dx = st.number_input("Duration Since Diagnosis (yrs)", 0, 50, 5)
            sleep_hours = st.number_input("Sleep Hours", 0, 15, 7)
            diet_quality = st.slider("Diet Quality (1=Poor, 10=Excellent)", 1, 10, 6)
            smoking_status = st.selectbox("Smoking Status (0=Non-smoker, 1=Smoker)", [0, 1])

        col3, col4 = st.columns(2)
        with col3:
            alcohol_units_week = st.number_input("Alcohol Units/Week", 0, 50, 2)
            med_adherence = st.slider("Medication Adherence (0=Poor, 10=Excellent)", 0, 10, 8)
            glucose_trend_3h = st.number_input("Glucose Trend Last 3h (mg/dL change)", -100, 100, 0)
        with col4:
            avg_glucose_7d = st.number_input("Average Glucose (last 7 days)", 50, 400, 150)
            time_since_last_insulin = st.number_input("Time Since Last Insulin (hrs)", 0, 48, 6)
            hyper_events_past_week = st.number_input("Hyperglycemia Events (past week)", 0, 10, 1)

        submit_btn = st.form_submit_button("Calculate Risk")

    # =========================
    # Prediction Logic
    # =========================
    if submit_btn:
        input_data = np.array([[
            glucose, carbs, insulin, activity, stress, time_of_day,
            age, bmi, hba1c, hypertension, kidney_disease, type_diabetes,
            duration_since_dx, sleep_hours, diet_quality, smoking_status,
            alcohol_units_week, med_adherence, glucose_trend_3h,
            avg_glucose_7d, time_since_last_insulin, hyper_events_past_week
        ]])

        input_scaled = scaler.transform(input_data)
        pred_probs = model.predict_proba(input_scaled)[0]
        pred_class = np.argmax(pred_probs)

        risk_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
        risk_colors = {"Low Risk": "green", "Moderate Risk": "orange", "High Risk": "red"}
        risk_label = risk_map.get(pred_class, "Unknown")

        suggestion_map = {
            "Low Risk": "Maintain current lifestyle, regular monitoring, follow-up every 3–6 months.",
            "Moderate Risk": "Review diet, increase physical activity, monitor glucose daily, consult physician within a month.",
            "High Risk": "Immediate review by healthcare provider, possible medication adjustment, frequent monitoring (4+ times/day)."
        }
        suggestion = suggestion_map.get(risk_label, "No suggestion available.")

        st.subheader("📊 Prediction Results")
        st.write(f"**Predicted Risk Category:** {risk_label}")
        st.write(f"**Prediction Confidence:** {pred_probs[pred_class]*100:.2f}%")
        st.markdown(f"**Clinical Suggestion:** {suggestion}")

        # =========================
        # Better Plotly Bar Chart
        # =========================
        fig = go.Figure()
        for i, risk in enumerate(["Low Risk", "Moderate Risk", "High Risk"]):
            fig.add_trace(go.Bar(
                x=[risk],
                y=[pred_probs[i]],
                text=[f"{pred_probs[i]*100:.1f}%"],
                textposition="auto",
                marker_color=risk_colors[risk],
                name=risk
            ))

        fig.update_layout(
            title="Risk Probability Distribution",
            yaxis_title="Probability",
            xaxis_title="Risk Category",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)
