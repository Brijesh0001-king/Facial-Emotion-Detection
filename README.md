# Facial-Emotion-Detection
CNN-based facial emotion recognition  Supports 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral  Emotion stability using temporal buffering  Smart confidence filtering to reduce flickering
  # 🎯 AI Vision Assistant – Real-Time Emotion, Age & Identity Recognition

A **resume-grade real-time computer vision system** built using **Python, OpenCV, Deep Learning, and Face Recognition**. The system detects **facial emotions**, estimates **age and birth year**, recognizes **known individuals**, understands **hand gestures**, performs **gesture-based arithmetic**, and interacts using **speech** — all via a standard laptop webcam.

---

## 🚀 Features

### 🧠 Emotion Detection (Deep Learning)

* CNN-based facial emotion recognition
* Supports **7 emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
* Temporal smoothing using short & long buffers for stable predictions
* Confidence-based filtering to reduce flicker

### 👤 Face Recognition & Identity Detection

* Recognizes known people using face embeddings
* Identifies individuals under varying lighting conditions
* Displays **Name, Age Group, and Birth Year**
* Graceful handling of unknown faces

### 🎂 Age Estimation & Birth Year Prediction

* Age estimation using **OpenCV DNN (Caffe)** model
* Converts predicted age bucket to an **approximate birth year**

### ✋ Hand Gesture Recognition

* Finger counting using **MediaPipe Hands**
* Robust real-time hand landmark detection

### ➕ Gesture-Based Addition

* Show fingers to input numbers
* Capture first and second numbers via keyboard control
* Automatic addition with **spoken output**

### 🗣️ Voice Interaction

* Emotion-aware spoken feedback
* Identity-aware greetings
* Controlled speech timing to avoid repetition

### 📷 Webcam Image Enhancement

* CLAHE-based contrast enhancement
* Sharpening filters to improve blurry camera feeds
* Improved detection accuracy on low-quality webcams

---

## 🛠️ Tech Stack

| Category         | Technology              |
| ---------------- | ----------------------- |
| Language         | Python 3                |
| Computer Vision  | OpenCV                  |
| Deep Learning    | TensorFlow / Keras      |
| Face Recognition | face_recognition (dlib) |
| Hand Tracking    | MediaPipe               |
| Age Detection    | OpenCV DNN (Caffe)      |
| Speech           | pyttsx3                 |
| Data             | JSON                    |

---

## 📂 Project Structure

```
AI-Vision-Assistant/
│
|
│   
|
│
├── people_data.json
├── emotion_model.hdf5
├── age_deploy.prototxt
├── age_net.caffemodel
├──faceEmotionDetection.py
└── README.md
```

---

## ⚙️ Installation

```bash
pip install opencv-python mediapipe tensorflow numpy pyttsx3 face-recognition
```

> ⚠️ **Windows Note**: `face-recognition` requires **CMake** and **Visual Studio Build Tools**.

---

## ▶️ How to Run

```bash
python faceEmotionDetection.py
```

### Controls

* `1` → Capture first number using hand gesture
* `2` → Capture second number using hand gesture
* `Q` → Quit application

---

## 📌 Use Cases

* Human–Computer Interaction (HCI)
* Smart Surveillance Systems
* AI-based Attendance Systems
* Gesture-controlled Interfaces
* Assistive AI Applications
* Academic & Research Projects

---

## ⚠️ Disclaimer

* Age and emotion predictions are **approximate** and depend on lighting and camera quality.
* Face recognition works best with clear frontal images.

---

## 👨‍💻 Author

**Brijesh Rajpara**
B.Sc. IT | AI & Computer Vision Enthusiast

---

## 🌱 Future Enhancements

* Auto face registration via voice
* Emotion analytics per user
* Secure face-based authentication
* Cloud database integration
* Mobile camera (IP camera) support

---

⭐ If you find this project useful, please consider giving it a star on GitHub!
