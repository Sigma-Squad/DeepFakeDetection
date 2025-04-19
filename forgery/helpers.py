import dlib
import cv2
import numpy as np
from skimage.transform import resize
import random
from scipy.spatial import procrustes

def setup_predictor_and_detector():
    predictor_path = "/content/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    return detector, predictor

def get_landmarks(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    points = [(p.x, p.y) for p in landmarks.parts()]
    return np.array(points, dtype=np.int32)

def extract_face(augmented_image, landmarks):
    """ Extract the face region from the image using landmarks. """
    hull = cv2.convexHull(landmarks)
    mask = np.zeros(augmented_image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    face_only = cv2.bitwise_and(augmented_image, augmented_image, mask=mask)

    return face_only, mask

def augment_image(image):

    alpha = np.random.uniform(0.55, 0.65)  # Contrast
    beta = np.random.randint(-30, 30)      # Brightness
    augmented_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # random color shift
    if random.random() < 0.5:
        augmented_image = augmented_image.astype(np.int32)
        augmented_image[:, :, 0] += np.random.randint(-10, 10)  # B
        augmented_image[:, :, 1] += np.random.randint(-10, 10)  # G
        augmented_image[:, :, 2] += np.random.randint(-10, 10)  # R
        augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)

    return augmented_image

def landmark_similarity(landmarks1, landmarks2):
    """ Calculate the similarity between two sets of landmarks. """
    if landmarks1.shape != landmarks2.shape:
        raise ValueError("Landmarks must have the same shape")

    # Procrustes analysis
    mtx1, mtx2, disparity = procrustes(landmarks1, landmarks2)
    return disparity

def extract_all_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)

    cap.release()
    return frames if frames else None