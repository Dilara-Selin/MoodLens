import yt_dlp
import cv2
import numpy as np
import joblib
from deepface import DeepFace
from keras.models import load_model
import time
import os
import speech_recognition as sr
from pydub import AudioSegment

# YouTube videosunu indirme fonksiyonu
def download_video(youtube_url, output_path="aysu_video.mp4"):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'quiet': False,
        'retries': 10,
        'no_warnings': True,
        'merge_output_format': 'mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    print("Video indirildi!")
    return output_path

# Video analiz fonksiyonu (yüz tanıma + duygu analizi)
def analyze_video(video_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    knn_model = joblib.load("face_knn_model.pkl")
    emotion_model = load_model("model_dropout.h5")

    cap = cv2.VideoCapture(video_path)

    fps_limit = 5
    interval = 1.0 / fps_limit
    last_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if (current_time - last_time) < interval:
            continue
        last_time = current_time

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            # 1. Yüz tanıma
            try:
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                result = DeepFace.represent(img_path=rgb_face, model_name="Facenet", enforce_detection=False)
                embedding = np.expand_dims(result[0]['embedding'], axis=0)

                prediction = knn_model.predict(embedding)
                name = prediction[0]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 100), 2)
                cv2.putText(frame, f"Name: {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

            except Exception as e:
                print(f"Yüz Tanıma Hatası: {e}")

            # 2. Duygu analizi
            try:
                resized_face = cv2.resize(face_img, (48, 48))
                resized_face = resized_face / 255.0
                resized_face = np.expand_dims(resized_face, axis=0)

                prediction = emotion_model.predict(resized_face)
                emotion_label = "Sad" if prediction[0][0] > 0.5 else "Happy"
                emotion_score = max(prediction[0][0], 1 - prediction[0][0]) * 100
                emotion_text = f"{emotion_label}: {emotion_score:.1f}%"

                cv2.putText(frame, emotion_text, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            except Exception as e:
                print(f"Duygu Analizi Hatası: {e}")

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def live_camera_analysis():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    knn_model = joblib.load("face_knn_model.pkl")
    emotion_model = load_model("model_dropout.h5")

    cap = cv2.VideoCapture(0)
    fps_limit = 5
    interval = 1.0 / fps_limit
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if (current_time - last_time) < interval:
            continue
        last_time = current_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            # Yüz Tanıma
            try:
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                result = DeepFace.represent(img_path=rgb_face, model_name="Facenet", enforce_detection=False)
                embedding = np.expand_dims(result[0]['embedding'], axis=0)

                prediction = knn_model.predict(embedding)
                name = prediction[0]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 100), 2)
                cv2.putText(frame, f"Name: {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

            except Exception as e:
                print(f"Yüz Tanıma Hatası: {e}")

            # Duygu Analizi
            try:
                resized_face = cv2.resize(face_img, (48, 48))
                resized_face = resized_face / 255.0
                resized_face = np.expand_dims(resized_face, axis=0)

                prediction = emotion_model.predict(resized_face)
                emotion_label = "Sad" if prediction[0][0] > 0.5 else "Happy"
                emotion_score = max(prediction[0][0], 1 - prediction[0][0]) * 100
                emotion_text = f"{emotion_label}: {emotion_score:.1f}%"

                cv2.putText(frame, emotion_text, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            except Exception as e:
                print(f"Duygu Analizi Hatası: {e}")

        cv2.imshow("Kamera - Canlı Analiz (Çıkmak için 'q' bas)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Yeni Fonksiyonlar ---

def extract_audio_from_video(video_path, output_audio_path="temp_audio.wav"):
    """
    Videodan sesi çıkarır ve wav formatında kaydeder.
    """
    from moviepy.editor import VideoFileClip

    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path, codec='pcm_s16le')
    audio.close()
    video.close()

    return output_audio_path

def transcribe_audio(audio_path):
    """
    SpeechRecognition kullanarak wav dosyasını yazıya çevirir ve toplam ses süresini dakika olarak verir.
    """
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(audio_path)

    with audio_file as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language="tr-TR")
    except sr.UnknownValueError:
        text = "[Konuşma anlaşılamadı]"
    except sr.RequestError as e:
        text = f"[Google API hatası: {e}]"

    # Toplam konuşma süresi (dakika)
    audio_seg = AudioSegment.from_wav(audio_path)
    duration_sec = audio_seg.duration_seconds
    duration_min = round(duration_sec / 60, 2)

    return text, duration_min

def identify_speaker_transcribe_and_emotion(video_path):
    """
    Videodan konuşan kişinin kim olduğunu, duygu analizini yapar,
    konuşma metnini transkribe eder ve toplam konuşma süresini dakika olarak verir.
    """

    # Yüz tanıma ve duygu analizi - ilk frame'den
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    knn_model = joblib.load("face_knn_model.pkl")
    emotion_model = load_model("model_dropout.h5")

    cap = cv2.VideoCapture(video_path)

    detected_name = None
    detected_emotion = None

    ret, frame = cap.read()
    if ret:
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            try:
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                result = DeepFace.represent(img_path=rgb_face, model_name="Facenet", enforce_detection=False)
                embedding = np.expand_dims(result[0]['embedding'], axis=0)

                prediction = knn_model.predict(embedding)
                detected_name = prediction[0]

            except Exception:
                detected_name = "Bilinmiyor"

            try:
                resized_face = cv2.resize(face_img, (48, 48))
                resized_face = resized_face / 255.0
                resized_face = np.expand_dims(resized_face, axis=0)

                prediction = emotion_model.predict(resized_face)
                detected_emotion = "Sad" if prediction[0][0] > 0.5 else "Happy"

            except Exception:
                detected_emotion = "Bilinmiyor"

    cap.release()

    # Ses çıkarma ve yazıya çevirme
    audio_path = extract_audio_from_video(video_path)
    transcription, duration_min = transcribe_audio(audio_path)

    os.remove(audio_path)  # Geçici ses dosyasını sil

    result_str = (
        f"Konuşan kişi: {detected_name}\n"
        f"Konuşma metni: {transcription}\n"
        f"Toplam konuşma süresi (dakika): {duration_min}\n"
        f"Duygu: {detected_emotion}"
    )
    return result_str
