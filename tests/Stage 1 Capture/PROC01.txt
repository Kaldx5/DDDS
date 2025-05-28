import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model("cnnCat2.h5")

# Capture one frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

# Convert to grayscale and resize to 24x24
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (24, 24))
normalized = resized / 255.0
input_data = normalized.reshape(1, 24, 24, 1)

# Predict
prediction = model.predict(input_data)
class_idx = np.argmax(prediction)
label = "Closed" if class_idx == 0 else "Open"

print(f"Prediction: {label} (Confidence: {prediction[0][class_idx]:.2f})")
