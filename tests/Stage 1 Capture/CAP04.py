import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("🎯 Starting Eye/Face Detection... Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            drawing.draw_landmarks(frame, face, mp_face_mesh.FACEMESH_CONTOURS)

    cv2.imshow('Eye/Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
