import tensorflow as tf
import keras
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pygame
import numpy as np

print("✅ TensorFlow version:", tf.__version__)
print("✅ Keras version:", keras.__version__)
print("✅ OpenCV version:", cv2.__version__)
print("✅ Mediapipe version:", mp.__version__)
print("✅ Matplotlib version:", plt.__version__)
print("✅ Pygame version:", pygame.__version__)

# Simple numpy array test
arr = np.array([[1, 2], [3, 4]])
print("✅ Numpy test array:\n", arr)

# Simple matplotlib plot
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("✅ Matplotlib Test Plot")
plt.show()

# Simple OpenCV image creation
img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.putText(img, 'OpenCV', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv2.imshow('✅ OpenCV Window', img)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# Simple Pygame window
pygame.init()
screen = pygame.display.set_mode((300, 200))
pygame.display.set_caption('✅ Pygame Window Test')
screen.fill((30, 144, 255))
pygame.display.flip()
pygame.time.wait(1000)
pygame.quit()
