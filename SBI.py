import dlib
import cv2
import numpy as np
from skimage.transform import resize
import random
import helpers as help

def SBI_face(image):
    detector, predictor = help.setup_predictor_and_detector()
    # Detect landmarks
    landmarks = help.get_landmarks(image, detector, predictor)
    if landmarks is None:
        return image
    augmented_image = help.augment_image(image)
    # Extract face region
    face_only, mask = help.extract_face(augmented_image, landmarks)

    return face_only, mask




