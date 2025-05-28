import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import pygame

# Initialize pygame for sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("../ddds_main/alarm.wav")

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# EAR + MAR helpers
def calculate_EAR(landmarks, eye_indices):
    eye = np.array([landmarks[i] for i in eye_indices])
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def calculate_MAR(landmarks, mouth_indices):
    mouth = np.array([landmarks[i] for i in mouth_indices])
    A = np.linalg.norm(mouth[1] - mouth[5])
    B = np.linalg.norm(mouth[2] - mouth[4])
    C = np.linalg.norm(mouth[0] - mouth[3])
    return (A + B) / (2.0 * C)

# Landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 81, 311, 78, 308]

# TFLite model
interpreter = tf.lite.Interpreter(model_path="../models/cnn_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Logging setup
log_file = open("../ddds_main/drowsiness_log.txt", "a")

# Video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

            leftEAR = calculate_EAR(landmarks, LEFT_EYE)
            rightEAR = calculate_EAR(landmarks, RIGHT_EYE)
            avgEAR = (leftEAR + rightEAR) / 2.0
            mar = calculate_MAR(landmarks, MOUTH)

            input_data = np.array([[[avgEAR], [mar]]], dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = output_data[0][0]

            label = "DROWSY" if prediction > 0.5 else "AWAKE"
            color = (0, 0, 255) if label == "DROWSY" else (0, 255, 0)

            # Alarm + log
            if label == "DROWSY":
                pygame.mixer.Sound.play(alarm_sound)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"Drowsiness detected at {timestamp}\n")
                log_file.flush()

            # Display
            cv2.putText(frame, f"{label} ({prediction:.2f})", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    cv2.imshow('Drowsiness Detector', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
