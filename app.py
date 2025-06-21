import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load model
model = joblib.load("decision_tree_model.pkl")

# Buat dummy data buat get_dummies
df_dummy = pd.read_csv("minuman_dataset.csv")
X_dummy = df_dummy[["waktu", "cuaca", "energi", "manis"]]
X_columns = pd.get_dummies(X_dummy).columns

# Mapping minuman → gambar
gambar_map = {
    "kopi": "images/kopi.png",
    "teh": "images/teh.jpg",
    "jus buah": "images/jus_buah.jpg",
    "coklat panas": "images/coklat_panas.jpg",
    "soda": "images/soda.jpg",
    "smoothie": "images/smoothie.jpg",
    "air mineral": "images/air_mineral.jpg",
    "es teh manis": "images/es_teh_manis.jpg",
}

# Sidebar judul
st.sidebar.title("Mood kamu hari ini?")

# Input user
waktu = st.sidebar.selectbox("Pilih waktu:", ["pagi", "siang", "sore", "malam"])
cuaca = st.sidebar.selectbox("Cuaca:", ["panas", "dingin", "hujan", "berawan"])
energi = st.sidebar.selectbox("Butuh energi:", ["tinggi", "sedang", "rendah"])
manis = st.sidebar.selectbox("Mau yang manis?", ["ya", "tidak"])

# Tombol rekomendasi
if st.sidebar.button("Rekomendasikan Minuman"):

    # Siapkan data user
    user_input = pd.DataFrame({
        "waktu": [waktu],
        "cuaca": [cuaca],
        "energi": [energi],
        "manis": [manis]
    })

    # One-hot encoding
    user_encoded = pd.get_dummies(user_input)
    user_encoded = user_encoded.reindex(columns=X_columns, fill_value=0)

    # Prediksi
    hasil = model.predict(user_encoded)[0]

    # Tampilkan hasil
    st.header(f"☕ Rekomendasi minuman: **{hasil.title()}**")

    # Tampilkan gambar
    gambar_path = gambar_map.get(hasil.lower())
    if gambar_path:
        image = Image.open(gambar_path)
        st.image(image, caption=hasil.title(), use_container_width=True)
    else:
        st.write("⚠️ Gambar belum tersedia.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Faisal")
