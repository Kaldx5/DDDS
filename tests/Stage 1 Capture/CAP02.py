import cv2
import time

cap = cv2.VideoCapture(0)
fps_list = []

if not cap.isOpened():
    print("❌ Camera failed to open.")
else:
    print("⏱ Measuring FPS... Press Q to stop.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_list.append(fps)

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("FPS Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Print average FPS
avg_fps = sum(fps_list) / len(fps_list)
print(f"📊 Average FPS: {avg_fps:.2f}")
