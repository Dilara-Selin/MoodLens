##
# @file moodlens_analysis.py
# @author Dilara Selin SALCI
# @brief Bu dosya, videolarda konuşan kişileri tanıyan, duygu durumlarını analiz eden
#        ve konuşma süresi ile metnini çıkaran bir analiz sistemini içerir.
#        DeepFace, KNN, CNN modelleri ve Google Speech Recognition kullanmaktadır.
##

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
from moviepy.editor import VideoFileClip

##
# @brief Belirtilen YouTube videosunu MP4 formatında indirir.
# @param youtube_url İndirilecek YouTube video URL’si.
# @param output_path Videonun kaydedileceği dosya adı (varsayılan: "aysu_video.mp4").
# @return Kaydedilen video dosyasının yolu.
##
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

##
# @brief Videodaki birden fazla yüzü tanır ve her biri için duygu analizi yapar.
# @param video_path Analiz edilecek video dosyasının yolu.
##
def analyze_video_multi_face(video_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    knn_model = joblib.load("face_knn_model.pkl")
    emotion_model = load_model("model_dropout.h5")

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            try:
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                result = DeepFace.represent(img_path=rgb_face, model_name="Facenet", enforce_detection=False)
                embedding = np.expand_dims(result[0]['embedding'], axis=0)

                prediction = knn_model.predict(embedding)
                name = prediction[0]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 255, 100), 2)
                cv2.putText(frame, f"Name: {name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

                resized_face = cv2.resize(face_img, (48, 48))
                resized_face = resized_face / 255.0
                resized_face = np.expand_dims(resized_face, axis=0)
                pred_emotion = emotion_model.predict(resized_face)
                emotion_label = "Sad" if pred_emotion[0][0] > 0.5 else "Happy"
                cv2.putText(frame, f"Emotion: {emotion_label}", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            except Exception as e:
                print(f"Hata: {e}")

        cv2.imshow("Multi Face Video Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

##
# @brief Canlı kamera akışında yüz tanıma ve duygu analizi yapar.
##
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

##
# @brief Videodan sesi çıkarır ve WAV formatında kaydeder.
# @param video_path İşlenecek video dosyasının yolu.
# @param output_audio_path Oluşturulacak ses dosyasının yolu (varsayılan: "temp_audio.wav").
# @return Oluşturulan ses dosyasının yolu.
##
def extract_audio_from_video(video_path, output_audio_path="temp_audio.wav"):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path, codec='pcm_s16le')
    audio.close()
    video.close()
    return output_audio_path

##
# @brief WAV dosyasını yazıya çevirir ve toplam ses süresini dakika olarak döndürür.
# @param audio_path Yazıya çevrilecek ses dosyasının yolu.
# @return (transcribed_text, duration_minute): Yazıya dökülmüş metin ve süresi (dakika cinsinden).
# @throws sr.UnknownValueError Konuşma anlaşılamadığında.
# @throws sr.RequestError Google API erişim hatası oluştuğunda.
##
def transcribe_audio(audio_path):
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

    audio_seg = AudioSegment.from_wav(audio_path)
    duration_sec = audio_seg.duration_seconds
    duration_min = round(duration_sec / 60, 2)

    return text, duration_min

##
# @brief Videodaki kişileri tanır, duygularını analiz eder, ses verisini yazıya çevirir ve süre analizini yapar.
# @param video_path Analiz edilecek video dosyasının yolu.
# @return Tanınan kişiler, duyguları, görünme süresi, konuşma metni ve ses süresini içeren detaylı rapor (metin formatında).
##
def identify_speaker_transcribe_and_emotion(video_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    knn_model = joblib.load("face_knn_model.pkl")
    emotion_model = load_model("model_dropout.h5")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    detected_faces = {}
    appearance_counts = {}

    frame_counter = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames_to_process = min(1000, total_frames)

    while frame_counter < max_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]

            try:
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                result = DeepFace.represent(img_path=rgb_face, model_name="Facenet", enforce_detection=False)
                embedding = np.expand_dims(result[0]['embedding'], axis=0)
                name = knn_model.predict(embedding)[0]
            except Exception:
                name = "Bilinmiyor"

            try:
                resized_face = cv2.resize(face_img, (48, 48))
                resized_face = resized_face / 255.0
                resized_face = np.expand_dims(resized_face, axis=0)
                prediction = emotion_model.predict(resized_face)
                emotion = "Sad" if prediction[0][0] > 0.5 else "Happy"
            except Exception:
                emotion = "Bilinmiyor"

            if name not in detected_faces:
                detected_faces[name] = emotion

            appearance_counts[name] = appearance_counts.get(name, 0) + 1

        frame_counter += 1

    cap.release()

    audio_path = extract_audio_from_video(video_path)
    transcription, duration_min = transcribe_audio(audio_path)
    os.remove(audio_path)

    result_lines = ["Görüntüde Tanınan Kişiler ve Duyguları:"]
    for name, emotion in detected_faces.items():
        result_lines.append(f"- {name}: {emotion}")

    result_lines.append("\nKişilerin Görünme Süresi:")
    for name, count in appearance_counts.items():
        duration_sec = round(count / fps, 2)
        result_lines.append(f"- {name}: {duration_sec} saniye ({count} kare)")

    result_lines.append("\nKonuşma Metni:")
    result_lines.append(transcription)
    result_lines.append(f"\nToplam Konuşma Süresi (dakika): {duration_min}")

    return "\n".join(result_lines)