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
    import time  # Yangi import
    
    known_face_ids, known_face_encodings = barcha_yuz_kodlarini_olish()
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Kamera ochib bo'lmadi! Iltimos, kamera ulanganligini tekshiring.")
        return
    
    # Kamera sozlamalari
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 1280 o'rniga 640
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 720 o'rniga 480
    video_capture.set(cv2.CAP_PROP_FPS, 15)  # 30 o'rniga 15
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = video_capture.read()
            if not ret:
                print("Kadrni o'qib bo'lmadi!")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Soddalashtirilgan preprocessing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Yuzlarni aniqlash (HOG modeli)
            face_locations = face_recognition.face_locations(
                rgb_frame, 
                model="hog",
                number_of_times_to_upsample=1
            )
            
            # Faqat bir nechta yuz uchun kodlash
            face_encodings = face_recognition.face_encodings(
                rgb_frame, 
                face_locations[:3],  # Faqat dastlabki 3 ta yuz
                num_jitters=1,  # 3 o'rniga 1
                model="small"  # large o'rniga small
            )
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                try:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        
                        if face_distances[best_match_index] < 0.6:  # 0.5 o'rniga 0.6
                            face_id = known_face_ids[best_match_index]
                            name = f"{face_id}-Jangchi"
                            kirishni_loglash(face_id)
                        else:
                            face_image = frame[top:bottom, left:right]
                            face_id = bazaga_yuz_qoshish(face_image, face_encoding)
                            if face_id:
                                name = f"Yangi {face_id}-Jangchi"
                                known_face_ids.append(face_id)
                                known_face_encodings.append(face_encoding)
                    else:
                        face_image = frame[top:bottom, left:right]
                        face_id = bazaga_yuz_qoshish(face_image, face_encoding)
                        if face_id:
                            name = f"Yangi {face_id}-Jangchi"
                            known_face_ids.append(face_id)
                            known_face_encodings.append(face_encoding)
                    
                    # Ekranga chizish
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left + 6, bottom - 6), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                except Exception as e:
                    print(f"Yuzni qayta ishlashda xatolik: {e}")
                    continue
            
            cv2.imshow('Yuzni tanib olish', frame)
            
            # Chiqish shartlari
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:  # ESC tugmasi
                break
                
            # FPS ni cheklash
            elapsed_time = time.time() - start_time
            if elapsed_time < 0.066:  # ~15 FPS
                time.sleep(0.066 - elapsed_time)
    
    except Exception as e:
        print(f"Kutilmagan xatolik: {e}")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Jadvallarni yaratish
    if create_tables():
        print("Tizim ishga tushmoqda...")
        yuzni_tanib_olish()
    else:
        print("Jadvallarni yaratishda xatolik. Dastur to'xtatildi.")