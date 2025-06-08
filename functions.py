import yt_dlp
import cv2
import numpy as np
import joblib
from deepface import DeepFace
from keras.models import load_model
import time

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
import cv2
import numpy as np
import joblib
from deepface import DeepFace
from keras.models import load_model
import time

def analyze_video(video_path, output_video_path="analyzed_output.mp4", log_path="loglar.txt"):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    knn_model = joblib.load("face_knn_model.pkl")
    emotion_model = load_model("model_dropout.h5")

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    skip_frames = 5  # Her 5. kareyi analiz edeceğiz
    results = []

    kişiler_süre = {}   # Kişi adına göre toplam süre (saniye)
    duygular_süre = {}  # Duygu adına göre toplam süre (saniye)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Analiz Başladı\n\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("🎬 Video bitti.")
            break

        frame_count += 1

        if frame_count % skip_frames != 0:
            out.write(frame)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            name = "Bilinmiyor"
            emotion_text = "Tespit edilemedi"
            emotion_label = None  # Duygu etiketi (Sad/Happy)

            # Yüz tanıma
            try:
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                result = DeepFace.represent(img_path=rgb_face, model_name="Facenet", enforce_detection=False)
                embedding = np.expand_dims(result[0]['embedding'], axis=0)
                prediction = knn_model.predict(embedding)
                name = prediction[0]
            except Exception as e:
                print(f"Yüz Tanıma Hatası: {e}")

            # Duygu analizi
            try:
                resized_face = cv2.resize(face_img, (48, 48))
                resized_face = resized_face / 255.0
                resized_face = np.expand_dims(resized_face, axis=0)
                prediction = emotion_model.predict(resized_face)
                emotion_label = "Sad" if prediction[0][0] > 0.5 else "Happy"
                emotion_score = max(prediction[0][0], 1 - prediction[0][0]) * 100
                emotion_text = f"{emotion_label} ({emotion_score:.1f}%)"
            except Exception as e:
                print(f"Duygu Analizi Hatası: {e}")

            # Kişi ve duygu sürelerini hesapla
            saniye = skip_frames / fps
            kişiler_süre[name] = kişiler_süre.get(name, 0) + saniye
            if emotion_label:
                duygular_süre[emotion_label] = duygular_süre.get(emotion_label, 0) + saniye

            # Video üzerine yazılar
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 100), 2)
            cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, emotion_text, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            # Log kaydı
            log_line = f"Kare #{frame_count} | Kişi: {name} | Duygu: {emotion_text}"
            print(log_line)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")

            results.append({
                "Kare": frame_count,
                "Kişi": name,
                "Duygu": emotion_text
            })

        out.write(frame)

    cap.release()
    out.release()

    # Sonuçları güzelce döndür
    özet = {
        "Toplam Kare": frame_count,
        "Kişi Bazında Toplam Süre (sn)": kişiler_süre,
        "Duygu Bazında Toplam Süre (sn)": duygular_süre,
        "Detaylı Sonuçlar": results
    }

    print(f"\n✅ İşlem tamamlandı! Çıkış videosu: {output_video_path}, Log dosyası: {log_path}")
    return özet


# Kameradan canlı analiz yapan fonksiyon
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
