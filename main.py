import cv2
import face_recognition
import psycopg2
import numpy as np
from datetime import datetime

# PostgreSQL ulanishi
conn = psycopg2.connect(
    dbname="face_db",
    user="postgres",
    password="2002",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Jadvalni yaratish (ushbu qismni qo'shing)
cursor.execute("""
CREATE TABLE IF NOT EXISTS face_data (
    id SERIAL PRIMARY KEY,
    person_id INTEGER,
    face_encoding BYTEA,
    image_data BYTEA,
    capture_time TIMESTAMP,
    person_name VARCHAR(100)
)
""")
conn.commit()

# Odamning ismini so'rash
person_name = input("Odamning ismini kiriting: ")

# Person_id uchun yangi ID generatsiya qilish
cursor.execute("SELECT COALESCE(MAX(person_id), 0) + 1 FROM face_data")
person_id = cursor.fetchone()[0]

# Kamera ochish
video_capture = cv2.VideoCapture(0)
saved_images = 0

while saved_images < 10:
    ret, frame = video_capture.read()
    
    # Yuzlarni topish
    face_locations = face_recognition.face_locations(frame)
    
    for face_location in face_locations:
        if saved_images >= 10:
            break
            
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        
        # Rasm o'lchamini standartlashtirish
        face_image = cv2.resize(face_image, (200, 200))
        
        face_encodings = face_recognition.face_encodings(frame, [face_location])
        if len(face_encodings) == 0:
            continue
            
        face_encoding = face_encodings[0]
        _, img_encoded = cv2.imencode('.jpg', face_image)
        
        # Ma'lumotlar bazasiga saqlash
        cursor.execute(
            """INSERT INTO face_data 
            (person_id, face_encoding, image_data, capture_time, person_name) 
            VALUES (%s, %s, %s, %s, %s)""",
            (person_id, 
             psycopg2.Binary(face_encoding.tobytes()), 
             psycopg2.Binary(img_encoded.tobytes()),
             datetime.now(),
             person_name)
        )
        conn.commit()
        
        saved_images += 1
        print(f"Saqlangan rasmlar: {saved_images}/10 - {person_name}")
        
        # Ekranda yuzni chizish
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, f"{person_name} {saved_images}/10", 
                   (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Natijani ko'rsatish
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Resurslarni ozod qilish
video_capture.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()