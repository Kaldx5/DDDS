from scipy.spatial import distance as dist
import numpy as np

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_EAR(landmarks, left_eye_indices=LEFT_EYE, right_eye_indices=RIGHT_EYE):
    left_eye = np.array([landmarks[i] for i in left_eye_indices])
    right_eye = np.array([landmarks[i] for i in right_eye_indices])
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    return (left_ear + right_ear) / 2.0
