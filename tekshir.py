import cv2
import face_recognition
import psycopg2
import numpy as np
import io
from datetime import datetime

# 🔌 PostgreSQL ga ulanish
conn = psycopg2.connect(
    host="localhost",
    database="face_db",
    user="postgres",
    password="2002"
)
cursor = conn.cursor()

# 📥 Bazadagi mavjud yuzlarni yuklash
def load_known_faces():
    cursor.execute("SELECT id, img FROM face_data")
    rows = cursor.fetchall()
    ids = []
    encodings = []

    for id_, img_bytes in rows:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces_enc = face_recognition.face_encodings(rgb)
        if faces_enc:
            ids.append(id_)
            encodings.append(faces_enc[0])

    return ids, encodings

# 🎥 Kamera ishga tushadi
cap = cv2.VideoCapture(0)
print("Kamera ishga tushdi... Yuzlar solishtirilmoqda.")

known_ids, known_encodings = load_known_faces()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan rasm olinmadi.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        face_id = None

        if True in matches:
            match_index = matches.index(True)
            face_id = known_ids[match_index]

            # ✅ Log yozish
            cursor.execute(
                "INSERT INTO face_log_data (id_name, kirish_vaqti) VALUES (%s, %s)",
                (face_id, datetime.now())
            )
            conn.commit()
            print(f"[TANILDI] Yuz ID: {face_id} - Kirish vaqti yozildi.")
        else:
            # ❌ Yangi yuz → rasm saqlanadi
            top, right, bottom, left = face_location
            new_face = frame[top:bottom, left:right]

            _, buffer = cv2.imencode('.jpg', new_face)
            img_bytes = buffer.tobytes()

            cursor.execute(
                "INSERT INTO face_data (img) VALUES (%s) RETURNING id",
                (psycopg2.Binary(img_bytes),)
            )
            new_id = cursor.fetchone()[0]

            # Log yozish
            cursor.execute(
                "INSERT INTO face_log_data (id_name, kirish_vaqti) VALUES (%s, %s)",
                (new_id, datetime.now())
            )
            conn.commit()

            # 📌 Ro‘yxatga yangi encodingni qo‘shish
            known_ids.append(new_id)
            known_encodings.append(face_encoding)

            print(f"[YANGI] Yangi yuz saqlandi, ID: {new_id}")

    # 🖼 Kamerani ko‘rsatish
    cv2.imshow("Kamera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🧹 Tozalash
cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
