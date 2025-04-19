import dlib
import cv2
import numpy as np
from skimage.transform import resize
import random
from scipy.spatial import procrustes

def setup_predictor_and_detector():
    predictor_path = "datfile"
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

def compute_similarity_transform(landmarks_src, landmarks_tgt):
    """
    Compute the similarity transform (rotation, scaling, and translation)
    that best aligns landmarks_src to landmarks_tgt.
    Returns the rotation matrix, translation vector, and scale factor.
    """
    # Compute center of mass for source and target landmarks
    src_center = np.mean(landmarks_src, axis=0)
    tgt_center = np.mean(landmarks_tgt, axis=0)

    # Compute scale factor based on distances between corresponding points
    src_dist = np.linalg.norm(landmarks_src - src_center, axis=1)
    tgt_dist = np.linalg.norm(landmarks_tgt - tgt_center, axis=1)
    scale = np.mean(tgt_dist) / np.mean(src_dist)

    # Calculate rotation matrix using SVD (Singular Value Decomposition)
    src_centered = landmarks_src - src_center
    tgt_centered = landmarks_tgt - tgt_center
    H = np.dot(src_centered.T, tgt_centered)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Compute translation vector
    t = tgt_center - scale * np.dot(R, src_center)

    return R, t, scale

def build_affine_matrix(R, t, scale):
    """
    Build the 2x3 affine transformation matrix for applying the similarity transform.
    """
    M = np.eye(2, 3)  # Create a 2x3 identity matrix
    M[:2, :2] = R * scale  # Scale and rotate
    M[:2, 2] = t  # Translation
    return M

def warp_face(img_src, M, target_size):
    """
    Warp the source face image using the affine matrix M to fit into the target frame.
    """
    return cv2.warpAffine(img_src, M, target_size)

def blend_face_into_target(target_img, face_img, warped_mask):
    """
    Blend the warped source face into the target image using the warped mask.
    """
    # Mask out the region of the target where the face will go
    target_face = cv2.bitwise_and(target_img, target_img, mask=cv2.bitwise_not(warped_mask))
    
    # Mask the warped face to only include face pixels
    masked_face = cv2.bitwise_and(face_img, face_img, mask=warped_mask)
    
    # Add the masked warped face to the target image
    blended_face = cv2.add(target_face, masked_face)
    
    return blended_face

def fit_face_to_target(source_face, source_landmarks, target_mask, target_landmarks, target_img):
    """
    Fit the source face into the target mask, aligning based on landmarks, and blend with target image.
    
    :param source_face: The face image with background removed, same size as the target image.
    :param source_landmarks: Landmark points for the source face.
    :param target_mask: Mask defining the region to place the source face (binary mask).
    :param target_landmarks: Landmark points for the target face.
    :param target_img: The target image where the face needs to be placed.
    
    :return: The target image with the source face transformed and blended in.
    """
    # Step 1: Compute the similarity transform (rotation, scaling, and translation)
    R, t, scale = compute_similarity_transform(source_landmarks, target_landmarks)
    
    # Step 2: Build the affine transformation matrix
    M = build_affine_matrix(R, t, scale)
    
    # Step 3: Warp the source face image to fit the target geometry
    img_warped = warp_face(source_face, M, target_img.shape[1::-1])
    
    # Step 4: Blend the warped face image into the target image using the target mask
    # (No need to warp the target mask since it's already in target space)
    blended_img = blend_face_into_target(target_img, img_warped, target_mask)
    
    return blended_img
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

