import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import threading
import pygame
import argparse
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio

parser = argparse.ArgumentParser()
parser.add_argument('--simulate', action='store_true', help='Run in simulation mode without TFLite model')
args = parser.parse_args()

pygame.mixer.init()
alarm_file = "../sounds/alarm.wav"

def play_alarm():
    pygame.mixer.music.load(alarm_file)
    pygame.mixer.music.play()

if not args.simulate:
    interpreter = tf.lite.Interpreter(model_path="../models/cnn_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]

log_file = "../ddds_main/drowsiness_log.txt"
alarm_on = False
ear_threshold = 0.21
mar_threshold = 0.6
consec_frames = 25
counter = 0

cap = cv2.VideoCapture(0)
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    state = "AWAKE"
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

            left_eye = [landmarks[i] for i in LEFT_EYE]
            right_eye = [landmarks[i] for i in RIGHT_EYE]
            mouth = [landmarks[i] for i in MOUTH]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth)

            drowsy_detected = False
            if args.simulate:
                if ear < ear_threshold:
                    counter += 1
                else:
                    counter = 0
                drowsy_detected = counter >= consec_frames
            else:
                input_data = np.array([[[ear], [mar]]], dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0][0]
                if output_data > 0.5:
                    counter += 1
                else:
                    counter = 0
                drowsy_detected = counter >= consec_frames

            state = "DROWSY" if drowsy_detected else "AWAKE"
            color = (0, 0, 255) if state == "DROWSY" else (0, 255, 0)

            cv2.putText(frame, f"State: {state}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"EAR: {ear:.2f}, MAR: {mar:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            if state == "DROWSY":
                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=play_alarm, daemon=True).start()
                    with open(log_file, 'a') as f:
                        f.write(f"DROWSINESS DETECTED at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            else:
                alarm_on = False
                pygame.mixer.music.stop()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
