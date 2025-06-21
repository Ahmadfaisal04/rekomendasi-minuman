import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load model
model = joblib.load("decision_tree_model.pkl")

# Dummy buat get_dummies
df_dummy = pd.read_csv("minuman_dataset.csv")
X_dummy = df_dummy[["waktu", "cuaca", "energi", "manis"]]
X_columns = pd.get_dummies(X_dummy).columns

import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load model
model = joblib.load("decision_tree_model.pkl")

# Dummy buat get_dummies
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

# Mapping minuman → penjelasan
penjelasan_map = {
    "kopi": "Memberikan dorongan energi yang pas untuk memulai atau melanjutkan hari.",
    "teh": "Menenangkan pikiran sambil menikmati suasana.",
    "jus buah": "Segar dan penuh vitamin, cocok untuk menjaga stamina.",
    "coklat panas": "Hangat dan manis, pilihan yang pas di cuaca dingin.",
    "soda": "Menyegarkan di tengah cuaca panas atau saat butuh semangat.",
    "smoothie": "Lezat dan sehat, cocok untuk penambah energi alami.",
    "air mineral": "Pilihan terbaik untuk menjaga hidrasi tubuh kapan saja.",
    "es teh manis": "Segar, manis, cocok dinikmati kapan pun kamu mau.",
}
st.set_page_config(page_title="Bike Sharing Analysis", layout="wide")
# --- Sidebar ---

logo_path = "https://i.ibb.co/b03zmS6/Logo-Etika-2025.png"
st.sidebar.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 8px;">
        <img src="{logo_path}" width="auto" height="80px">
        <span style="font-size: 20px; font-weight: bold;">ETIKA STUDIO</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("## Mood kamu hari ini?")

# --- Input user ---
waktu = st.sidebar.selectbox("Pilih waktu:", ["pagi", "siang", "sore", "malam"])
cuaca = st.sidebar.selectbox("Cuaca:", ["panas", "dingin", "hujan", "berawan"])
energi = st.sidebar.selectbox("Butuh energi:", ["tinggi", "sedang", "rendah"])
manis = st.sidebar.selectbox("Mau yang manis?", ["ya", "tidak"])

# --- Tombol rekomendasi ---
if st.sidebar.button("Rekomendasikan Minuman 🚀"):

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

    # Tampilkan penjelasan
    penjelasan = penjelasan_map.get(hasil.lower(), "Minuman yang cocok untuk kamu hari ini!")
    st.markdown(
        f"""
        <div style="font-size: 16px; color: #555; margin-top: 12px;">
            💡 {penjelasan}
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 14px;">
        Made with ❤️ by Faisal
    </div>
    """,
    unsafe_allow_html=True
)
