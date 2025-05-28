import cv2
import numpy as np
from keras.models import load_model
import time
from datetime import datetime

# Load trained model
model = load_model("cnnCat2.h5")

cap = cv2.VideoCapture(0)

closed_frames = 0
fps_estimate = 28  # Use your measured FPS
closed_threshold = int(2 * fps_estimate)  # 2 seconds worth of frames

log = []  # Store events
alert_triggered = False

print("🚦 Starting real-time detection... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (24, 24))
    normalized = resized / 255.0
    input_data = normalized.reshape(1, 24, 24, 1)

    prediction = model.predict(input_data)
    class_idx = np.argmax(prediction)
    label = "Closed" if class_idx == 0 else "Open"
    confidence = prediction[0][class_idx]

    timestamp = datetime.now().strftime("%H:%M:%S")

    if label == "Closed":
        closed_frames += 1
        if closed_frames == 1:
            log.append(f"[{timestamp}] Eyes CLOSED started")
    else:
        if closed_frames >= closed_threshold:
            log.append(f"[{timestamp}] Eyes OPENED after drowsy state (Closed for {closed_frames} frames)")
        elif closed_frames > 0:
            log.append(f"[{timestamp}] Eyes OPENED (Closed for {closed_frames} frames)")
        closed_frames = 0
        alert_triggered = False

    if closed_frames >= closed_threshold and not alert_triggered:
        status = "ALERT: DROWSINESS DETECTED!"
        color = (0, 0, 255)
        log.append(f"[{timestamp}] 🚨 ALERT triggered after {closed_frames} frames closed")
        alert_triggered = True
    else:
        status = f"{label} ({confidence:.2f})"
        color = (0, 255, 0)

    # Draw status on screen
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Closed frames: {closed_frames}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print log summary
print("\n📋 Event Log:")
for entry in log:
    print(entry)

# Optional: Save to text file
with open("drowsiness_log.txt", "w") as f:
    for entry in log:
        f.write(entry + "\n")
