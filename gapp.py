import streamlit as st
from gfunctions import download_video, identify_speaker_transcribe_and_emotion, live_camera_analysis
import tempfile

st.title("Video ve Canlı Kamera Duygu Analizi Uygulaması")

# Video URL ile indirme ve analiz
video_url = st.text_input("Video URL'sini gir ve indir")

if video_url:
    with st.spinner("Video indiriliyor... Sabırlı ol güzelim ❤️"):
        video_path = download_video(video_url)
        st.success("Video indirildi!")
        st.video(video_path)

        if st.button("İndirilen videoyu analiz et"):
            with st.spinner("Analiz yapılıyor..."):
                result_text = identify_speaker_transcribe_and_emotion(video_path)
                st.success("Analiz tamamlandı!")
                st.text(result_text)

st.markdown("---")

# Bilgisayardan video yükleme ve analiz
uploaded_file = st.file_uploader("Bilgisayardan video yükle", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Yüklenen videoyu analiz et"):
        with st.spinner("Analiz yapılıyor... Sabırlı ol güzelim ❤️"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_video_path = tmp_file.name

            result_text = identify_speaker_transcribe_and_emotion(tmp_video_path)
            st.success("Analiz tamamlandı!")
            st.text(result_text)

st.markdown("---")

# Canlı kamera analizi butonu
if st.button("Canlı Kamera Analizini Başlat"):
    st.write("Canlı kamera analizi başlatılıyor... Çıkmak için video penceresinde 'q' tuşuna bas.")
    live_camera_analysis()
    st.write("Canlı analiz sonlandı.")
