import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# EAR calculation helper
def calculate_EAR(landmarks, eye_indices):
    # Get eye points
    eye = np.array([landmarks[i] for i in eye_indices])
    # Compute distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Eye landmark indices (Mediapipe)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Thresholds
EAR_THRESH = 0.23
EAR_CONSEC_FRAMES = 15
COUNTER = 0

# Start video
cap = cv2.VideoCapture(0)
prev_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip + RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

            # Calculate EAR
            leftEAR = calculate_EAR(landmarks, LEFT_EYE)
            rightEAR = calculate_EAR(landmarks, RIGHT_EYE)
            avgEAR = (leftEAR + rightEAR) / 2.0

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            # Check if eyes closed
            if avgEAR < EAR_THRESH:
                COUNTER += 1
                if COUNTER >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSY ALERT!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            else:
                COUNTER = 0

            # Display EAR
            cv2.putText(frame, f"EAR: {avgEAR:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Drowsiness Detector', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
