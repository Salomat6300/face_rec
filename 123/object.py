import cv2
from ultralytics import YOLO

# YOLO modelini yuklaymiz
model = YOLO("yolov5s.pt")  # Bu fayl avtomatik yuklanadi (yoki yolov8n.pt bo'lishi mumkin)

# Kamerani ishga tushiramiz (0 — default webcam)
cap = cv2.VideoCapture(0)

# Kamera ochilmagan bo‘lsa chiqish
if not cap.isOpened():
    print("Kamera ochilmadi!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Modelga tasvirni uzatamiz
    results = model.predict(frame, imgsz=640, conf=0.3)[0]

    # Har bir aniqlangan obyektni chizamiz
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])     # Koordinatalar
        conf = float(box.conf[0])                  # Ishonchlilik
        cls_id = int(box.cls[0])                   # Class ID
        label = model.names[cls_id]                # Obyekt nomi

        # To‘g‘ri to‘rtburchak chizish
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Obyekt nomini yuqoriga yozish
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Natijaviy oynani chiqaramiz
    cv2.imshow("Real-Time Object Detection", frame)

    # ESC tugmasi bosilsa dastur to‘xtaydi
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Resurslarni bo‘shatamiz
cap.release()
cv2.destroyAllWindows()
