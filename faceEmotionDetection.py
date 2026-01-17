import cv2
import numpy as np
import pyttsx3
import time
from collections import deque
from tensorflow.keras.models import load_model
from datetime import datetime
import mediapipe as mp

# ================= LOAD MODELS =================
emotion_model = load_model("emotion_model.hdf5", compile=False)

age_net = cv2.dnn.readNetFromCaffe(
    "age_deploy (1).prototxt",
    "age_net.caffemodel"
)

# ================= CONSTANTS =================
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
AGE_BUCKETS = ['0-2','4-6','8-12','15-20','21-30','31-43','44-53','54+']

# ================= SPEECH =================
engine = pyttsx3.init()
engine.setProperty('rate', 150)
speech_enabled = True

def speak(text):
    if speech_enabled:
        engine.say(text)
        engine.runAndWait()

# ================= FACE =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= HANDS =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def count_fingers(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = 0

    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0]-1].x:
        fingers += 1

    for tip in tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
            fingers += 1

    return fingers

# ================= BUFFERS =================
short_buffer = deque(maxlen=5)
long_buffer = deque(maxlen=20)
last_spoken_emotion = None
last_speak_time = 0

# ================= IMAGE ENHANCEMENT =================
def enhance_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(gray, -1, kernel)

# ================= PREPROCESS =================
def preprocess_face(face):
    face = cv2.resize(face, (64,64))
    face = face.astype("float32") / 255.0
    return face.reshape(1,64,64,1)

# ================= AGE =================
def estimate_age(face_bgr):
    blob = cv2.dnn.blobFromImage(
        face_bgr, 1.0, (227,227),
        (78.42, 87.76, 114.89),
        swapRB=False
    )
    age_net.setInput(blob)
    preds = age_net.forward()
    return AGE_BUCKETS[preds[0].argmax()]

def birth_year_from_bucket(bucket):
    start_age = int(bucket.replace('+','').split('-')[0])
    return datetime.now().year - start_age

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
print("SYSTEM RUNNING | Hand Gestures Enabled | Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = enhance_gray(gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion = "Unknown"
    age_group = "N/A"
    birth_year = "N/A"

    for (x,y,w,h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        face_bgr = frame[y:y+h, x:x+w]

        preds = emotion_model.predict(preprocess_face(face_gray), verbose=0)[0]

        preds[1] *= 1.3
        preds[2] *= 1.2
        preds[5] *= 1.2

        short_buffer.append(preds)
        long_buffer.append(preds)

        avg = np.mean(short_buffer, axis=0)
        label = np.argmax(avg)
        confidence = np.max(avg)
        emotion = emotion_labels[label]

        age_group = estimate_age(face_bgr)
        birth_year = birth_year_from_bucket(age_group)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,f"{emotion} ({int(confidence*100)}%)",(x,y-30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)

        cv2.putText(frame,f"Age: {age_group} | Birth: ~{birth_year}",
                    (x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
        break

    # ================= HAND GESTURE CONTROL =================
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            fingers = count_fingers(hand)

            cv2.putText(frame,f"Fingers: {fingers}",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

            if fingers == 2:
                speak(f"You look {emotion}")

            elif fingers == 5:
                speak(f"Your age group is {age_group}. Birth year around {birth_year}")

            elif fingers == 1:
                speech_enabled = False

            elif fingers == 0:
                speak("System shutting down")
                cap.release()
                cv2.destroyAllWindows()
                exit()

            break

    cv2.imshow("Professional AI Vision System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

