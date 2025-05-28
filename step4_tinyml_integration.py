import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
drawing_utils = mp.solutions.drawing_utils

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

def dummy_tinyml_predict(ear):
    return "DROWSY" if ear < 0.2 else "AWAKE"

cap = cv2.VideoCapture(0)

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

            status = dummy_tinyml_predict(avg_ear)
            cv2.putText(frame, f"Status: {status}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

            drawing_utils.draw_landmarks(
                frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    cv2.imshow('TinyML Placeholder', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
