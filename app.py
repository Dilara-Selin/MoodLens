"""
@file app.py
@brief Streamlit arayüzü ile video ve canlı kamera üzerinden duygu analizi yapan uygulama.

Bu uygulama YouTube videosu ya da bilgisayardan yüklenen videoları analiz eder
ve canlı kamera üzerinden gerçek zamanlı duygu tespiti gerçekleştirir.

@author Dilara Selin SALCI
@date 2025-06-08
@version 1.0
"""

import streamlit as st
from functions import download_video, analyze_video, live_camera_analysis
import tempfile

"""
@mainpage MoodLens - Duygu Analizi Arayüzü
Bu ana sayfa, kullanıcıdan video veya canlı görüntü alarak yapay zeka destekli duygu tespiti yapar.
"""

st.title("🎥 Video & Kamera Duygu Analizi")

# YouTube videosu
video_url = st.text_input("🎬 YouTube video linkini buraya yapıştır güzelim:")

if video_url:
    with st.spinner("Videoyu indiriyorum... Bekle biraz ❤️"):
        video_path = download_video(video_url)
        st.success("Video başarıyla indirildi!")
        st.video(video_path)

        if st.button("Bu videoyu analiz et"):
            with st.spinner("Analiz ediliyor..."):
                results = analyze_video(video_path)
                st.success("Analiz tamamlandı!")
                st.write(results)

st.markdown("---")

# Bilgisayardan video
uploaded_file = st.file_uploader("💻 Bilgisayardan video yükle (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Yüklediğim videoyu analiz et"):
        with st.spinner("Videoyu analiz ediyorum..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_video_path = tmp_file.name

            results = analyze_video(tmp_video_path)
            st.success("Analiz tamamlandı!")
            st.write(results)

st.markdown("---")

# Canlı kamera
if st.button("📷 Canlı Kamerayı Başlat"):
    st.info("Canlı kamera açılıyor... Çıkmak için video penceresinde 'q' tuşuna bas.")
    live_camera_analysis()