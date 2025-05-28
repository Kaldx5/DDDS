import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Resize to model input size (example: 224x224)
    resized = cv2.resize(gray, (224, 224))

    # 3. Show processed frame
    cv2.imshow("Grayscale Preprocessed Frame", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
