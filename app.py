import streamlit as st
import pandas as pd
import numpy as np
import joblib

# KONFIGURASI HALAMAN (Wajib di baris pertama setelah import)
st.set_page_config(
    page_title="Health Risk AI",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# LOAD MODEL & ASSETS
@st.cache_resource
def load_assets():
    # Menggunakan decision.pkl sesuai permintaan
    model = joblib.load("decision.pkl")
    scaler = joblib.load("scaler.pkl")
    le_gender = joblib.load("labelencoder_gender.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, scaler, le_gender, feature_cols

try:
    model, scaler, le_gender, feature_cols = load_assets()
except FileNotFoundError as e:
    st.error(f"File tidak ditemukan: {e}. Pastikan file .pkl sudah diupload.")
    st.stop()

# CUSTOM CSS UI
st.markdown("""
    <style>
        /* Mengubah font utama */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }

        /* Styling Header */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .main-header h1 {
            color: white;
            font-weight: 700;
            margin-bottom: 10px;
            font-size: 2.5rem;
        }
        .main-header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        /* Styling Form & Container */
        .stButton button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px;
            border: none;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        /* Styling Result Card */
        .result-box {
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            margin-top: 20px;
            box-shadow: 0 10px 15px rgba(0,0,0,0.1);
            animation: fadeIn 0.8s;
        }
        .safe {
            background-color: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        .danger {
            background-color: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("Tentang Aplikasi")
    st.write("""
    Aplikasi ini menggunakan **Decision Tree Classifier** untuk memprediksi risiko kesehatan berdasarkan gaya hidup dan metrik medis Anda.
    
    **Fitur Utama:**
    - Analisis BMI & Tekanan Darah
    - Deteksi faktor gaya hidup
    - Prediksi real-time
    """)
    st.markdown("---")
    st.caption("© 2024 Health Risk AI Team")

# HEADER UTAMA
st.markdown("""
    <div class="main-header">
        <h1>🩺 Prediksi Risiko Kesehatan</h1>
        <p>Analisis Cerdas untuk Gaya Hidup Lebih Sehat</p>
    </div>
""", unsafe_allow_html=True)

# CONTAINER INPUT UTAMA
with st.container():
    st.info("ℹ️ Silakan isi data di bawah ini dengan lengkap untuk hasil yang akurat.")
    
    with st.form("health_form"):
        # Kelompokkan input agar lebih rapi
        
        # BAGIAN 1: PROFIL PRIBADI
        st.subheader("👤 Profil Pribadi")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Umur (Tahun)", 1, 100, 30)
        with c2:
            gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        with c3:
            family_history = st.selectbox("Riwayat Penyakit Keluarga?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")

        st.markdown("---")

        # BAGIAN 2: GAYA HIDUP
        st.subheader("🏃 Gaya Hidup & Kebiasaan")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            daily_steps = st.slider("Langkah Harian", 0, 20000, 6000, help="Rata-rata langkah kaki per hari")
            water_intake_l = st.number_input("Asupan Air (Liter)", 0.0, 10.0, 2.0, step=0.1)
            sleep_hours = st.slider("Durasi Tidur (Jam)", 0.0, 12.0, 7.0, step=0.5)
            calories_consumed = st.number_input("Kalori Harian", 500, 5000, 2200)
        
        with col_l2:
            smoker = st.radio("Apakah Anda Perokok?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak", horizontal=True)
            alcohol = st.radio("Konsumsi Alkohol?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak", horizontal=True)

        st.markdown("---")

        # BAGIAN 3: METRIK KESEHATAN
        st.subheader("❤️ Metrik Medis")
        with st.expander("Klik untuk input data medis detail", expanded=True):
            m1, m2 = st.columns(2)
            with m1:
                bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0)
                resting_hr = st.number_input("Detak Jantung Istirahat (BPM)", 40, 150, 75)
                cholesterol = st.number_input("Level Kolesterol Total", 100, 500, 180)
            with m2:
                systolic_bp = st.number_input("Tekanan Darah Sistolik", 80, 250, 120)
                diastolic_bp = st.number_input("Tekanan Darah Diastolik", 50, 150, 80)

        submit_btn = st.form_submit_button("🔍 Analisis Sekarang")

# LOGIC PREDIKSI
if submit_btn:
    with st.spinner("Sedang menganalisis data kesehatan Anda..."):
        try:
            # 1. Encode Gender
            gender_encoded = le_gender.transform([gender])[0]

            # 2. Feature Engineering (Sesuai logika training)
            bp_ratio = systolic_bp / diastolic_bp if diastolic_bp != 0 else 0
            pulse_pressure = systolic_bp - diastolic_bp
            is_obese = int(bmi >= 30)
            low_sleep = int(sleep_hours < 6)
            risk_score = smoker + alcohol + is_obese + low_sleep

            # 3. Build DataFrame
            input_data = {
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
            }
            
            user_df = pd.DataFrame([input_data])
            
            # Pastikan urutan kolom sesuai dengan training
            user_df = user_df[feature_cols]

            # 4. Scaling
            user_scaled = scaler.transform(user_df)

            # 5. Prediksi
            prediction = model.predict(user_scaled)[0]
            probs = model.predict_proba(user_scaled)[0]
            
            # Ambil probabilitas kelas positif (1)
            risk_percent = probs[1] * 100

            # TAMPILAN HASIL
            st.markdown("### 📊 Hasil Analisis")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if prediction == 1:
                    st.markdown(f"""
                        <div class="result-box danger">
                            <h1>⚠️</h1>
                            <h3>BERISIKO TINGGI</h3>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-box safe">
                            <h1>✅</h1>
                            <h3>RISIKO RENDAH</h3>
                        </div>
                    """, unsafe_allow_html=True)

            with col_res2:
                st.write("#### Probabilitas Risiko:")
                st.progress(int(risk_percent))
                st.caption(f"Tingkat kemungkinan risiko terdeteksi: **{risk_percent:.2f}%**")
                
                if prediction == 1:
                    st.warning("Saran: Segera konsultasikan dengan dokter untuk pemeriksaan lebih lanjut dan perbaiki gaya hidup Anda.")
                else:
                    st.success("Saran: Pertahankan gaya hidup sehat Anda! Tetap pantau kesehatan secara rutin.")

            # Tampilkan data mentah dalam expander (opsional)
            with st.expander("Lihat Data Input"):
                st.dataframe(user_df)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")
