import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("mnist_model.h5")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Define region of interest box
    x1, y1, x2, y2 = 300, 100, 450, 250
    roi = frame[y1:y2, x1:x2]

    # Preprocess the ROI to match MNIST style
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi_thresh = cv2.threshold(roi_gray, 120, 255, cv2.THRESH_BINARY_INV)

    roi_resized = cv2.resize(roi_thresh, (28, 28))
    roi_normalized = roi_resized.astype("float32") / 255.0
    roi_input = roi_normalized.reshape(1, 28, 28, 1)

    # Predict digit using GPU
    pred = model.predict(roi_input, verbose=0)
    digit = np.argmax(pred)

    # Draw UI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f"Prediction: {digit}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Digit Recognition (GPU Powered)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
