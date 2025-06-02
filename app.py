import streamlit as st
from PIL import Image
import numpy as np
import joblib
from ekstraksi import ekstraksi_fitur_dari_gambar

# Load model
rf_model = joblib.load("rf_model.pkl")
svm_model = joblib.load("svm_model.pkl")

st.title("Klasifikasi Kematangan Pisang dengan Streamlit")

uploaded_file = st.file_uploader("Upload gambar pisang (jpg/png/jpeg)", type=["jpg", "png", "jpeg"])

model_choice = st.selectbox("Pilih model klasifikasi:", ("Random Forest", "SVM"))

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    label_prediksi = None

    if st.button("Prediksi Kematangan"):
        # Ekstraksi fitur otomatis tanpa label
        fitur = ekstraksi_fitur_dari_gambar(img)

        fitur_array = np.array([list(fitur.values())])

        if model_choice == "Random Forest":
            pred = rf_model.predict(fitur_array)[0]
            proba = rf_model.predict_proba(fitur_array)[0]
            classes = rf_model.classes_
        else:
            pred = svm_model.predict(fitur_array)[0]
            proba = svm_model.predict_proba(fitur_array)[0]
            classes = svm_model.classes_

        st.success(f"Prediksi kelas: **{pred}**")
        st.write("Probabilitas per kelas:")
        for cls, p in zip(classes, proba):
            st.write(f"- {cls}: {p:.2%}")

