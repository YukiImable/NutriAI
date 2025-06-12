import streamlit as st
import numpy as np
import pandas as pd
import json
import joblib
from tensorflow.keras.models import load_model

# Load semua file yang diperlukan 
model_paths = {
    "Logistic Regression": "model/model_logistic_regression.h5",
    "Neural Network": "model/model_neural_network.h5",
    "Wide & Deep": "model/model_wide_and_deep.h5"
}

scaler = joblib.load("model/scaler.pkl")

with open("model/feature_order.json") as f:
    feature_order = json.load(f)

with open("model/numerical_features.json") as f:
    numerical_features = json.load(f)

# Konfigurasi Streamlit 
st.set_page_config(page_title="NutriAI", layout="wide")
st.title("🍁 NutriAI - Form Prediksi Diabetes")
st.markdown("<h3>Silakan lengkapi data pasien di bawah ini :</h3>", unsafe_allow_html=True)

#  Input Form 
# Demografi
age = st.number_input("Usia", 20, 90, 30, key="age")
gender = st.radio("Jenis Kelamin", [0, 1], format_func=lambda x: "Laki-laki" if x == 0 else "Perempuan", key="gender")
ethnicity = st.selectbox("Etnis", [0, 1, 2, 3], key="ethnicity")
socio = st.selectbox("Status Sosial Ekonomi", [0, 1, 2], key="socio")
edu = st.selectbox("Tingkat Pendidikan", [0, 1, 2, 3], key="edu")

# Gaya Hidup
bmi = st.number_input("BMI", 15.0, 40.0, key="bmi")
smoking = st.radio("Merokok?", [0, 1], key="smoking")
alcohol = st.slider("Konsumsi Alkohol per Minggu", 0.0, 20.0, key="alcohol")
activity = st.slider("Aktivitas Fisik (jam/minggu)", 0.0, 10.0, key="activity")
diet = st.slider("Kualitas Diet (0–10)", 0.0, 10.0, key="diet")
sleep = st.slider("Kualitas Tidur (4–10)", 4.0, 10.0, key="sleep")

# Riwayat Medis
family = st.radio("Riwayat Keluarga Diabetes?", [0, 1], key="family")
gestational = st.radio("Diabetes Gestasional?", [0, 1], key="gestational")
pcos = st.radio("PCOS?", [0, 1], key="pcos")
pre_diabetes = st.radio("Pre-Diabetes?", [0, 1], key="pre_diabetes")
htn = st.radio("Hipertensi?", [0, 1], key="htn")

# Pengukuran Klinis
sys_bp = st.number_input("Tekanan Darah Sistolik", 90, 180, key="sys_bp")
dia_bp = st.number_input("Tekanan Darah Diastolik", 60, 120, key="dia_bp")
fbs = st.number_input("Gula Darah Puasa (mg/dL)", 70.0, 200.0, key="fbs")
hba1c = st.number_input("HbA1c (%)", 4.0, 10.0, key="hba1c")
creatinine = st.number_input("Kreatinin Serum (mg/dL)", 0.5, 5.0, key="creatinine")
bun = st.number_input("BUN (mg/dL)", 5.0, 50.0, key="bun")
chol_total = st.number_input("Kolesterol Total (mg/dL)", 150.0, 300.0, key="chol_total")
ldl = st.number_input("Kolesterol LDL (mg/dL)", 50.0, 200.0, key="ldl")
hdl = st.number_input("Kolesterol HDL (mg/dL)", 20.0, 100.0, key="hdl")
trig = st.number_input("Trigliserida (mg/dL)", 50.0, 400.0, key="trig")

# Obat
obat_htn = st.radio("Obat Antihipertensi?", [0, 1], key="obat_htn")
statin = st.radio("Gunakan Statin?", [0, 1], key="statin")
obat_dm = st.radio("Obat Diabetes?", [0, 1], key="obat_dm")

# Gejala
urin = st.radio("Sering Buang Air Kecil?", [0, 1], key="urin")
haus = st.radio("Haus Berlebihan?", [0, 1], key="haus")
weightloss = st.radio("Berat Turun Tanpa Sebab?", [0, 1], key="weightloss")
fatigue = st.slider("Tingkat Kelelahan (0–10)", 0.0, 10.0, key="fatigue")
blur = st.radio("Penglihatan Kabur?", [0, 1], key="blur")
slow_heal = st.radio("Luka Sulit Sembuh?", [0, 1], key="slow_heal")
tingle = st.radio("Kesemutan Tangan/Kaki?", [0, 1], key="tingle")
qol = st.slider("Skor Kualitas Hidup (0–100)", 0.0, 100.0, key="qol")

# Lingkungan
metal = st.radio("Paparan Logam Berat?", [0, 1], key="metal")
chemical = st.radio("Paparan Bahan Kimia?", [0, 1], key="chemical")
water = st.radio("Kualitas Air Buruk?", [0, 1], key="water")

# Perilaku
checkup = st.slider("Check-up per Tahun", 0.0, 4.0, key="checkup")
adherence = st.slider("Kepatuhan Minum Obat (0–10)", 0.0, 10.0, key="adherence")
literacy = st.slider("Literasi Kesehatan (0–10)", 0.0, 10.0, key="literacy")

# Pilih Model 
model_choice = st.selectbox("Pilih Model Prediksi", list(model_paths.keys()))

# Submit Button 
if st.button("SUBMIT"):
    try:
        # Buat dictionary input
        input_data = {
            "Age": age, "Gender": gender, "BMI": bmi, "Smoking": smoking,
            "AlcoholIntake": alcohol, "PhysicalActivity": activity, "DietQuality": diet, "SleepQuality": sleep,
            "FamilyHistory": family, "GestationalDiabetes": gestational, "PCOS": pcos,
            "PreDiabetes": pre_diabetes, "Hypertension": htn,
            "SystolicBP": sys_bp, "DiastolicBP": dia_bp,
            "FastingBloodSugar": fbs, "HbA1c": hba1c,
            "Creatinine": creatinine, "BUN": bun,
            "TotalCholesterol": chol_total, "LDL": ldl, "HDL": hdl, "Triglycerides": trig,
            "AntihypertensiveMeds": obat_htn, "StatinUse": statin, "DiabetesMeds": obat_dm,
            "FrequentUrination": urin, "ExcessiveThirst": haus,
            "UnexplainedWeightLoss": weightloss, "FatigueLevel": fatigue,
            "BlurredVision": blur, "SlowHealing": slow_heal, "Tingling": tingle,
            "QualityOfLife": qol, "HeavyMetalExposure": metal,
            "ChemicalExposure": chemical, "PoorWaterQuality": water,
            "AnnualCheckups": checkup, "MedicationAdherence": adherence,
            "HealthLiteracy": literacy
        }

        input_df = pd.DataFrame([input_data])

        # One-hot encoding fitur kategorikal
        for i in range(1, 4):
            input_df[f"Ethnicity_{i}"] = int(ethnicity == i)
            input_df[f"EducationLevel_{i}"] = int(edu == i)
        for i in range(1, 3):
            input_df[f"SocioeconomicStatus_{i}"] = int(socio == i)

        # Pastikan semua kolom yang dibutuhkan tersedia
        for col in feature_order:
            if col not in input_df.columns:
                input_df[col] = 0

        # Urutkan kolom sesuai urutan fitur model
        input_df = input_df[feature_order]

        # Normalisasi fitur numerik
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # Load dan prediksi dengan model
        model = load_model(model_paths[model_choice])
        pred = model.predict(input_df)[0][0]
        label = int(pred >= 0.5)

        # Tampilkan hasil prediksi
        st.success(f"Hasil Prediksi: {'💉 Diabetes' if label == 1 else '✅ Tidak Diabetes'} (Probabilitas: {pred:.2f})")

        # ======== Rekomendasi jika hasil positif ========
        if label == 1:
            centroids = joblib.load("model/kmeans_centroids.pkl")
            numerical_features_rec = joblib.load("model/numerical_features_rec.pkl")

            from sklearn.metrics.pairwise import euclidean_distances

            # Mapping nama fitur dari training ke input
            mapping = {
                "AlcoholConsumption": "AlcoholIntake",
                "SerumCreatinine": "Creatinine",
                "BUNLevels": "BUN",
                "CholesterolTotal": "TotalCholesterol",
                "CholesterolLDL": "LDL",
                "CholesterolHDL": "HDL",
                "CholesterolTriglycerides": "Triglycerides",
                "FatigueLevels": "FatigueLevel",
                "QualityOfLifeScore": "QualityOfLife",
                "MedicalCheckupsFrequency": "AnnualCheckups"
            }
            rec_features = [mapping.get(f, f) for f in numerical_features_rec]

            # Deskripsi fitur untuk ditampilkan ke user
            deskripsi_fitur = {
                "BMI": "Indeks Massa Tubuh (BMI)",
                "FastingBloodSugar": "Gula darah puasa",
                "HbA1c": "Kadar HbA1c",
                "PhysicalActivity": "Aktivitas fisik (jam/minggu)",
                "DietQuality": "Skor kualitas diet",
                "SleepQuality": "Skor kualitas tidur",
                "SystolicBP": "Tekanan darah sistolik",
                "DiastolicBP": "Tekanan darah diastolik",
                "TotalCholesterol": "Kolesterol total",
                "LDL": "Kolesterol LDL",
                "HDL": "Kolesterol HDL",
                "Triglycerides": "Trigliserida",
                "Creatinine": "Kreatinin serum",
                "BUN": "BUN (Blood Urea Nitrogen)",
                "AlcoholIntake": "Konsumsi alkohol",
                "MedicationAdherence": "Kepatuhan minum obat",
                "HealthLiteracy": "Literasi kesehatan",
                "AnnualCheckups": "Frekuensi check-up per tahun",
                "FatigueLevel": "Tingkat kelelahan",
                "QualityOfLife": "Skor kualitas hidup"
            }

            # Aturan logika validasi rekomendasi
            logic_rules = {
                "BMI": lambda curr, rec: curr > rec,
                "FastingBloodSugar": lambda curr, rec: curr > rec,
                "HbA1c": lambda curr, rec: curr > rec,
                "LDL": lambda curr, rec: curr > rec,
                "HDL": lambda curr, rec: curr < rec,
                "Triglycerides": lambda curr, rec: curr > rec,
                "Creatinine": lambda curr, rec: curr > rec,
                "BUN": lambda curr, rec: curr > rec,
                "AlcoholIntake": lambda curr, rec: curr > rec,
                "PhysicalActivity": lambda curr, rec: curr < rec,
                "DietQuality": lambda curr, rec: curr < rec,
                "SleepQuality": lambda curr, rec: curr < rec,
                "AnnualCheckups": lambda curr, rec: curr < rec,
                "MedicationAdherence": lambda curr, rec: curr < rec,
                "HealthLiteracy": lambda curr, rec: curr < rec,
                "QualityOfLife": lambda curr, rec: curr < rec,
                "FatigueLevel": lambda curr, rec: curr > rec,
                "SystolicBP": lambda curr, rec: curr > rec,
                "DiastolicBP": lambda curr, rec: curr > rec,
                "TotalCholesterol": lambda curr, rec: curr > rec
            }

            # Fungsi pembuat rekomendasi dari perbandingan centroid
            def generate_recommendation(input_row, centroids, feature_names, threshold=0.05):
                available_features = [f for f in feature_names if f in input_row.columns]
                x = input_row[available_features].values.reshape(1, -1)
                centroids_filtered = centroids[:, [input_row.columns.get_loc(f) for f in available_features]]
                dists = euclidean_distances(x, centroids_filtered)
                centroid = centroids_filtered[dists.argmin()]
                recs = {}
                for i, feat in enumerate(available_features):
                    curr = x[0, i]
                    rec = centroid[i]
                    if abs(rec - curr) > threshold:
                        rule = logic_rules.get(feat)
                        if rule and not rule(curr, rec):
                            continue
                        recs[feat] = (curr, rec)
                return recs

            # Kembalikan nilai asli fitur numerik sebelum normalisasi
            original = input_df.copy()
            original[numerical_features] = scaler.inverse_transform(original[numerical_features])

            # Dapatkan rekomendasi
            recs = generate_recommendation(original.iloc[[0]], centroids, rec_features)

            # Tampilkan rekomendasi
            if recs:
                st.subheader("📝 Rekomendasi Personalisasi untuk Perbaikan Kesehatan:")
                for feat, (curr, rec) in recs.items():
                    desc = deskripsi_fitur.get(feat, feat)
                    st.write(f"- **{desc}**: Nilai saat ini = `{curr:.2f}`, target disarankan = `{rec:.2f}`")
            else:
                st.info("✅ Tidak ada rekomendasi spesifik. Nilai Anda sudah mendekati profil sehat.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")