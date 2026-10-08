import cv2
import numpy as np
import pyttsx3
import time
import threading

# Load YOLOv4-Tiny Model
weights = "yolov4-tiny.weights"
config = "yolov4-tiny.cfg"
net = cv2.dnn.readNet(weights, config)

# Load COCO class labels
types = "coco.names"
with open(types, "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Initialize TTS
engine = pyttsx3.init()

# Set Indonesian voice if available
voices = engine.getProperty('voices')
for voice in voices:
    if "indonesia" in voice.name.lower() or "indonesian" in voice.name.lower() or "id_" in voice.id.lower():
        engine.setProperty('voice', voice.id)
        break

# Function to speak asynchronously
speaking = False
def speak(message):
    global speaking
    if not speaking:
        speaking = True
        def _speak():
            engine.say(message)
            engine.runAndWait()
            time.sleep(1)
            speaking = False
        threading.Thread(target=_speak).start()

# Initialize Camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    boxes, confidences, class_ids = [], [], []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.6:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            label = CLASSES[class_ids[i]]
            confidence = confidences[i]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Estimate distance and speak asynchronously
            if w > 100 and not speaking:
                speak(f"Ada {label} di depan!")

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()