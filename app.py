import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD MODEL AND DATA
model = joblib.load("/decision.pkl")
scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("labelencoder_gender.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# CUSTOM CSS
st.markdown("""
    <style>
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: #2E86C1;
            text-align: center;
            padding-bottom: 10px;
        }
        .sub-title {
            font-size: 20px;
            text-align: center;
            color: #555;
            margin-bottom: 30px;
        }
        .form-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .result-card {
            padding: 30px;
            border-radius: 12px;
            color: white;
            margin-top: 30px;
            text-align: center;
        }
        .low-risk {
            background-color: #2ECC71;
        }
        .high-risk {
            background-color: #E74C3C;
        }
        .input-container {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }
        .input-container .stTextInput, .input-container .stNumberInput, .input-container .stSelectbox {
            padding: 10px;
            font-size: 16px;
        }
        .stButton {
            font-size: 18px;
            padding: 10px;
            background-color: #2E86C1;
            color: white;
            border-radius: 8px;
        }
        .stButton:hover {
            background-color: #1C6F92;
        }
        .stSelectbox, .stNumberInput {
            font-size: 16px;
            padding: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# TITLE
st.markdown('<div class="main-title">🩺 Health Risk Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Masukkan data kesehatan Anda untuk memprediksi risiko penyakit</div>', unsafe_allow_html=True)

# INPUT FORM
with st.form("input_form"):
    with st.container():
        st.markdown("### Informasi Pribadi dan Kesehatan")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Umur", 1, 120, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
            daily_steps = st.number_input("Langkah Harian", 0, 30000, 6000)
            sleep_hours = st.number_input("Jam Tidur", 0.0, 15.0, 7.0)
            water_intake_l = st.number_input("Asupan Air (liter/hari)", 0.0, 10.0, 2.0)
            calories_consumed = st.number_input("Kalori Harian", 800, 6000, 2200)

        with col2:
            resting_hr = st.number_input("Resting Heart Rate", 40, 150, 75)
            systolic_bp = st.number_input("Systolic BP", 80, 220, 120)
            diastolic_bp = st.number_input("Diastolic BP", 50, 130, 80)
            cholesterol = st.number_input("Level Kolesterol", 100, 400, 180)
            family_history = st.selectbox("Riwayat Keluarga Penyakit?", [0, 1])
            smoker = st.selectbox("Perokok?", [0, 1])
            alcohol = st.selectbox("Konsumsi Alkohol?", [0, 1])

        submit = st.form_submit_button("🔍 Prediksi Risiko")

# PREDICTION
if submit:
    gender_encoded = le_gender.transform([gender])[0]

    # Feature engineering
    bp_ratio = systolic_bp / diastolic_bp
    pulse_pressure = systolic_bp - diastolic_bp
    is_obese = int(bmi >= 30)
    low_sleep = int(sleep_hours < 6)
    risk_score = smoker + alcohol + is_obese + low_sleep

    # Build dataframe
    user_input = pd.DataFrame([{
        "age": age,
        "gender": gender_encoded,
        "bmi": bmi,
        "daily_steps": daily_steps,
        "sleep_hours": sleep_hours,
        "water_intake_l": water_intake_l,
        "calories_consumed": calories_consumed,
        "smoker": smoker,
        "alcohol": alcohol,
        "resting_hr": resting_hr,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol": cholesterol,
        "family_history": family_history,
        "bp_ratio": bp_ratio,
        "pulse_pressure": pulse_pressure,
        "is_obese": is_obese,
        "low_sleep": low_sleep,
        "risk_score": risk_score
    }])

    # Pastikan urutan kolom benar
    user_input = user_input[feature_cols]

    # Scale the features
    user_scaled = scaler.transform(user_input)

    pred = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0][1] * 100  # dalam persen

    # OUTPUT CARD
    if pred == 1:
        st.markdown(
            f'<div class="result-card high-risk">'
            f'<h3>⚠️ Risiko Tinggi</h3>'
            f'<p>Probabilitas: <b>{prob:.2f}%</b></p>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-card low-risk">'
            f'<h3>✅ Risiko Rendah</h3>'
            f'<p>Probabilitas: <b>{prob:.2f}%</b></p>'
            '</div>',
            unsafe_allow_html=True
        )

    # tampilkan data input
    st.write("### Data yang Anda Masukkan")
    st.dataframe(user_input)
