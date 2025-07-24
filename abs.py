import cv2
import face_recognition
import numpy as np
from datetime import datetime
import psycopg2
from psycopg2 import sql
import io
import os

# Bazaga ulanish parametrlari
DB_NAME = "face_db"
DB_USER = "postgres"
DB_PASSWORD = "2002"
DB_HOST = "localhost"
DB_PORT = "5432"

def get_db_connection():
    """PostgreSQL bazasiga ulanish"""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        print(f"Bazaga ulanib bo'lmadi: {e}")
        return None

def create_tables():
    """Jadvallarni yaratish"""
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        cur = conn.cursor()
        
        # face_data jadvali
        cur.execute("""
        CREATE TABLE IF NOT EXISTS face_data (
            id SERIAL PRIMARY KEY,
            img BYTEA NOT NULL,
            encoding FLOAT[] NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # face_log_data jadvali
        cur.execute("""
        CREATE TABLE IF NOT EXISTS face_log_data (
            id SERIAL PRIMARY KEY,
            id_name INTEGER REFERENCES face_data(id),
            kirish_vaqti TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        print("Jadvallar muvaffaqiyatli yaratildi!")
        return True
    except Exception as e:
        print(f"Jadvallarni yaratishda xatolik: {e}")
        return False
    finally:
        if conn:
            conn.close()

def bazaga_yuz_qoshish(face_image, face_encoding):
    """Yangi yuzni bazaga qo'shish"""
    conn = get_db_connection()
    if conn is None:
        return None
    
    try:
        cur = conn.cursor()
        
        # Rasmni byte ko'rinishiga o'tkazish
        is_success, buffer = cv2.imencode(".jpg", face_image)
        io_buf = io.BytesIO(buffer)
        
        # Kodni list ko'rinishiga o'tkazish
        encoding_list = face_encoding.tolist()
        
        # face_data jadvaliga qo'shish
        cur.execute(
            sql.SQL("""
            INSERT INTO face_data (img, encoding) 
            VALUES (%s, %s)
            RETURNING id
            """),
            (io_buf.read(), encoding_list)
        )
        
        face_id = cur.fetchone()[0]
        
        # face_log_data jadvaliga kirishni yozish
        cur.execute(
            sql.SQL("""
            INSERT INTO face_log_data (id_name) 
            VALUES (%s)
            """),
            (face_id,)
        )
        
        conn.commit()
        return face_id
    except Exception as e:
        print(f"Yuzni bazaga qo'shishda xatolik: {e}")
        return None
    finally:
        if conn:
            conn.close()

def barcha_yuz_kodlarini_olish():
    """Bazadagi barcha yuz kodlarini olish"""
    conn = get_db_connection()
    if conn is None:
        return [], []
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, encoding FROM face_data")
        rows = cur.fetchall()
        
        known_face_ids = []
        known_face_encodings = []
        
        for row in rows:
            face_id = row[0]
            encoding_array = row[1]
            known_face_ids.append(face_id)
            known_face_encodings.append(np.array(encoding_array))
        
        return known_face_ids, known_face_encodings
    except Exception as e:
        print(f"Yuz kodlarini olishda xatolik: {e}")
        return [], []
    finally:
        if conn:
            conn.close()

def kirishni_loglash(face_id):
    """Kirishni logga yozish"""
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        cur = conn.cursor()
        cur.execute(
            sql.SQL("""
            INSERT INTO face_log_data (id_name) 
            VALUES (%s)
            """),
            (face_id,)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Kirishni loglashda xatolik: {e}")
        return False
    finally:
        if conn:
            conn.close()

def yuzni_tanib_olish():
    """Asosiy yuzni tanib olish funksiyasi"""
    # Bazadan ma'lum yuzlarni yuklash
    known_face_ids, known_face_encodings = barcha_yuz_kodlarini_olish()
    
    # Kamerani ishga tushirish
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Kamerani ochib bo'lmadi!")
        return
    
    while True:
        # Kameradan kadr olish
        ret, frame = video_capture.read()
        if not ret:
            print("Kadrni o'qib bo'lmadi!")
            break
        
        # Rasmni RGB formatiga o'tkazish
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Kadrdagi yuzlarni topish
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Yuzlarni kodlash
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Yuzni ma'lum yuzlar bilan solishtirish
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                
                name = "Noma'lum"
                face_id = None
                
                # Agar mos yuz topilsa
                if True in matches:
                    first_match_index = matches.index(True)
                    face_id = known_face_ids[first_match_index]
                    name = f"Shaxs {face_id}"
                    
                    # Kirishni logga yozish
                    kirishni_loglash(face_id)
                else:
                    # Yangi yuzni bazaga qo'shish
                    face_image = frame[top:bottom, left:right]
                    face_id = bazaga_yuz_qoshish(face_image, face_encoding)
                    if face_id:
                        name = f"Yangi shaxs {face_id}"
                        
                        # Ma'lum yuzlar ro'yxatini yangilash
                        known_face_ids.append(face_id)
                        known_face_encodings.append(face_encoding)
                
                # Yuz atrofida to'rtburchak chizish
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                
                # Yuz ostiga nom yozish
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            
            # Natijani ekranga chiqarish
            cv2.imshow('Yuzni tanib olish', frame)
            
        except Exception as e:
            print(f"Yuzni tanib olishda xatolik: {e}")
            continue
        
        # Chiqish uchun 'q' tugmasini bosing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kamerani qo'yib yuborish
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Jadvallarni yaratish
    if create_tables():
        print("Tizim ishga tushmoqda...")
        yuzni_tanib_olish()
    else:
        print("Jadvallarni yaratishda xatolik. Dastur to'xtatildi.")