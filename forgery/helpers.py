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




"""

delaunay morphing

def rect_contains(rect, point):
    return (rect[0] <= point[0] < rect[0]+rect[2]) and (rect[1] <= point[1] < rect[1]+rect[3])



def calculate_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)

    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))

    triangleList = subdiv.getTriangleList()
    delaunay_tri = []

    for t in triangleList:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        for i in range(3):
            for j in range(len(points)):
                if abs(pts[i][0] - points[j][0]) < 1.0 and abs(pts[i][1] - points[j][1]) < 1.0:
                    idx.append(j)
        if len(idx) == 3:
            delaunay_tri.append((idx[0], idx[1], idx[2]))

    return delaunay_tri



def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect, t2_rect, t2_rect_int = [], [], []

    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    size = (r2[2], r2[3])

    if r1[2] > 0 and r1[3] > 0 and r2[2] > 0 and r2[3] > 0 and len(t1_rect) == 3 and len(t2_rect) == 3:
        warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
        img2_rect = cv2.warpAffine(img1_rect, warp_mat, size, None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        img2_rect = img2_rect * mask

        if r2[1] + r2[3] <= img2.shape[0] and r2[0] + r2[2] <= img2.shape[1]:
            dst_area = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
            dst_area_copy = dst_area.copy()
            dst_area_copy = dst_area_copy * (1.0 - mask) + img2_rect
            img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst_area_copy
            return True

    return False



def warp_face_delaunay(src_img, dst_img, src_landmarks, dst_landmarks):
    h, w = dst_img.shape[:2]
    warped_img = np.copy(dst_img).astype(np.float32)

    if src_landmarks is None or dst_landmarks is None:
        return False

    if len(src_landmarks) != len(dst_landmarks) or len(src_landmarks) < 3:
        return False

    rect = (0, 0, w, h)

    try:
        dt = calculate_delaunay_triangles(rect, dst_landmarks)
        if not dt:
            return False

        for tri_indices in dt:
            t1 = [src_landmarks[i] for i in tri_indices]
            t2 = [dst_landmarks[i] for i in tri_indices]

            if len(t1) == 3 and len(t2) == 3:
                warp_triangle(src_img, warped_img, t1, t2)

    except Exception:
        return False

    return np.clip(warped_img, 0, 255).astype(np.uint8)

"""

