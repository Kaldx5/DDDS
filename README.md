# Driver Drowsiness Detection System (DDDS)

> 🎓 **Final Grade: AA – Computer Engineering Graduation Project 2024–2025**  
> Developed and presented at Istanbul Okan University  
> Supervised by: Asst. Prof. Emel Koç

A real-time drowsiness detection system powered by TinyML and computer vision, designed to monitor driver alertness using a webcam and low-power embedded devices.

---

## 🧠 Project Summary

The Driver Drowsiness Detection System (DDDS) detects fatigue symptoms such as prolonged eye closure and yawning using:
- **Facial Landmark Detection** (via MediaPipe)
- **Eye Aspect Ratio (EAR) & Mouth Aspect Ratio (MAR)**
- **TinyML model (TensorFlow Lite)** for classification
- **Audio and Visual Alerts** to help prevent accidents
- **Event Logging** for timestamped records

---

## 📷 System Overview

- **Input**: Live webcam or ESP32-CAM feed  
- **Processing**: EAR/MAR or ML inference  
- **Output**: Real-time status display, alarm audio, event log

---

## ⚙️ Features

- Real-time facial landmark detection  
- EAR & MAR threshold logic  
- Optional ML model inference via TensorFlow Lite  
- Alarm sound via Pygame  
- Event log file with timestamps

---

## 🧮 Architecture Components

| Module            | Description                                      |
|-------------------|--------------------------------------------------|
| `main.py`         | Entry point; coordinates the full pipeline       |
| `EAR.py` / `MAR.py` | Aspect ratio calculators for eye/mouth         |
| `face_mesh`       | Powered by MediaPipe for landmark detection      |
| `tflite_inference.py` | Optional ML-based drowsiness classifier      |
| `soundLog.py`     | Alarm control and logging handler                |

---

## 📸 Screenshots

![State Overlay](DDDS/drowsy.png)  
*EAR/MAR display and drowsy state overlay*

---

## 📋 References

- Alajlan, N. N., & Ibrahim, D. M. (2023). "DDD TinyML: A TinyML-Based Driver Drowsiness Detection Model Using Deep Learning." *Sensors*, 23(12), 1–20. https://www.mdpi.com/1424-8220/23/12/5696  
- TensorFlow Lite: https://www.tensorflow.org/lite  
- MediaPipe FaceMesh: https://google.github.io/mediapipe/solutions/face_mesh  
- Khaled Alrefai, Computer Engineering SRS – Driver Drowsiness Detection System 
- Khaled Alrefai, Software Design Document (SDD) – Driver Drowsiness Detection System 
---

## 👨‍💻 Author

**Khaled Alrefai**  
Computer Engineering – Istanbul Okan University  
Graduated with Final Grade: **AA**  
Supervisor: Asst. Prof. Emel Koç

---

## 📝 License

This project is licensed under the MIT License.
