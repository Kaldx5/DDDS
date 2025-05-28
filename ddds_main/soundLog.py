import cv2
import mediapipe as mp
import numpy as np
import time
import sounddevice as sd

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# EAR calculation helper
def calculate_EAR(landmarks, eye_indices):
    eye = np.array([landmarks[i] for i in eye_indices])
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Play beep sound
def beep(frequency=440, duration=0.5, fs=44100):
    t = np.linspace(0, duration, int(fs * duration), False)
    tone = np.sin(frequency * 2 * np.pi * t)
    sd.play(tone, fs)
    sd.wait()

# Eye indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Thresholds
EAR_THRESH = 0.23
EAR_CONSEC_FRAMES = 15
COUNTER = 0
drowsy_events = 0
start_time = time.time()

# Logging
log_file = open("drowsiness_log.txt", "w")

# Video
cap = cv2.VideoCapture(0)
prev_time = time.time()

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

            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            if avgEAR < EAR_THRESH:
                COUNTER += 1
                if COUNTER >= EAR_CONSEC_FRAMES:
                    drowsy_events += 1
                    cv2.putText(frame, "DROWSY ALERT!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                    beep()
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"Drowsiness detected at {timestamp}\n")
                    log_file.flush()
            else:
                COUNTER = 0

    # Session stats
    elapsed_min = (time.time() - start_time) / 60
    rate = drowsy_events / elapsed_min if elapsed_min > 0 else 0

    # Overlay metrics
    cv2.putText(frame, f"EAR: {avgEAR:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Events: {drowsy_events}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Rate/min: {rate:.1f}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Session: {int(elapsed_min)} min", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Drowsiness Detector', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
