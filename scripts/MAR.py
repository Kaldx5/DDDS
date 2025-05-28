from scipy.spatial import distance as dist
import numpy as np

MOUTH = [61, 291, 81, 178, 13, 14, 312, 308, 324, 318, 402, 317]

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # top-bottom
    B = dist.euclidean(mouth[4], mouth[8])   # top-bottom
    C = dist.euclidean(mouth[0], mouth[6])   # left-right
    mar = (A + B) / (2.0 * C)
    return mar

def calculate_MAR(landmarks, mouth_indices=MOUTH):
    mouth = np.array([landmarks[i] for i in mouth_indices])
    mar = mouth_aspect_ratio(mouth)
    return mar
