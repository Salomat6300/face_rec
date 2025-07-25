import cv2
import mediapipe as mp
import numpy as np

# MediaPipe sozlamalari
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Kamerani ishga tushirish
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    face_orientation = "No face"

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Ko‘z va burun koordinatalari
        left_eye = landmarks[33]   # Chap ko‘z tashqi cheti
        right_eye = landmarks[263] # O‘ng ko‘z tashqi cheti
        nose = landmarks[1]        # Burun uchi

        # Framega moslashtirish (piksel o‘lchamiga)
        left_eye_x = left_eye.x * w
        right_eye_x = right_eye.x * w
        left_eye_y = left_eye.y * h
        right_eye_y = right_eye.y * h
        nose_x = nose.x * w
        nose_y = nose.y * h

        # Ko‘zlar orasidagi gorizontal masofa (eye_width)
        eye_width = abs(right_eye_x - left_eye_x)

        # ——— YAW tekshiruvi (burun o‘rta markazda bo‘lishi kerak) ———
        eye_x_center = (left_eye_x + right_eye_x) / 2
        nose_x_offset = abs(nose_x - eye_x_center)

        yaw_ok = nose_x_offset < eye_width * 0.08  # ~8% gacha og‘ish normal

        # ——— PITCH tekshiruvi (burun va ko‘zlar vertikal masofa) ———
        eye_y_avg = (left_eye_y + right_eye_y) / 2
        vertical_dist = abs(nose_y - eye_y_avg)

        vertical_ratio = vertical_dist / eye_width
        pitch_ok = 0.3 < vertical_ratio < 0.6  # Frontal uchun normal nisbati

        # ——— Yekun natija ———
        if yaw_ok and pitch_ok:
            face_orientation = "True"
        else:
            face_orientation = "False"

        # Ekranga natijani yozish
        color = (0, 255, 0) if face_orientation == "True" else (0, 0, 255)
        cv2.putText(frame, f"Face Forward: {face_orientation}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Yordamchi (debug) ma’lumotlar
        cv2.putText(frame, f"Yaw OK: {yaw_ok}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Pitch OK: {pitch_ok}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Vert Ratio: {vertical_ratio:.2f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)

    else:
        # Yuz topilmasa
        cv2.putText(frame, "No face detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Oynani ko‘rsatish
    cv2.imshow("Face Orientation Detection", frame)

    # ESC bosilsa chiqish
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
