import cv2

cap = cv2.VideoCapture(0)  # 0 for default camera

if not cap.isOpened():
    print("❌ Camera failed to open.")
else:
    print("✅ Camera initialized. Streaming...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break
        cv2.imshow("Live Feed", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
