from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import face_recognition
import numpy as np
import time
import psycopg2
from psycopg2 import sql
import os
from datetime import datetime

app = FastAPI()
templates = Jinja2Templates(directory="templates")

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

video_capture = cv2.VideoCapture(0)
known_face_ids, known_face_encodings = barcha_yuz_kodlarini_olish()

def gen_frames():
    """Video kadrlarini generatsiya qilish"""
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        
        # Rasmni aylantirish va formatini o'zgartirish
        frame = cv2.flip(frame, 1)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Yuzlarni aniqlash
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Kordinatalarni asl o'lchamga qaytarish
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Yuzlarni solishtirish
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Noma'lum"
            face_id = None
            
            if True in matches:
                # Ma'lum yuzni tanib olish
                first_match_index = matches.index(True)
                face_id = known_face_ids[first_match_index]
                name = f"{face_id}-Foydalanuvchi"
                kirishni_loglash(face_id)
            
            # Yuz atrofida chiziq chizish
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Yuz ostiga yozuv qo'yish uchun fonda chiziq
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        # Rasmni JPEG formatiga o'tkazish
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Asosiy sahifa"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    """Video oqimi"""
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stats")
async def get_stats():
    """Statistik ma'lumotlarni olish"""
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Bazaga ulanib bo'lmadi")
    
    try:
        cur = conn.cursor()
        
        # Umumiy foydalanuvchilar soni
        cur.execute("SELECT COUNT(*) FROM face_data")
        total_users = cur.fetchone()[0]
        
        # Bugungi kirishlar soni
        cur.execute("""
        SELECT COUNT(*) FROM face_log_data 
        WHERE DATE(kirish_vaqti) = CURRENT_DATE
        """)
        today_entries = cur.fetchone()[0]
        
        # Oxirgi 5 kirish
        cur.execute("""
        SELECT f.id, l.kirish_vaqti 
        FROM face_log_data l
        JOIN face_data f ON l.id_name = f.id
        ORDER BY l.kirish_vaqti DESC
        LIMIT 5
        """)
        last_entries = cur.fetchall()
        
        return {
            "total_users": total_users,
            "today_entries": today_entries,
            "last_entries": [{"id": row[0], "time": row[1].strftime("%Y-%m-%d %H:%M:%S")} for row in last_entries]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ma'lumotlarni olishda xatolik: {e}")
    finally:
        if conn:
            conn.close()