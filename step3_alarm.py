import cv2
import mediapipe as mp
import pygame

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
drawing_utils = mp.solutions.drawing_utils

pygame.mixer.init()
pygame.mixer.music.load('alarm.wav')  # Add alarm.wav in same folder

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, eye_points):
    left = landmarks[eye_points[0]]
    right = landmarks[eye_points[3]]
    top = (landmarks[eye_points[1]].y + landmarks[eye_points[2]].y) / 2
    bottom = (landmarks[eye_points[4]].y + landmarks[eye_points[5]].y) / 2
    width = abs(left.x - right.x)
    height = abs(top - bottom)
    return height / width

cap = cv2.VideoCapture(0)
drowsy_frames = 0
threshold_frames = 20

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            landmark_list = landmarks.landmark

            left_ear = eye_aspect_ratio(landmark_list, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmark_list, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2

            if avg_ear < 0.2:
                drowsy_frames += 1
                if drowsy_frames >= threshold_frames:
                    cv2.putText(frame, "DROWSINESS ALERT!", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play(-1)  # Loop alarm
            else:
                drowsy_frames =_
