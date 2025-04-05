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