import dlib
import cv2
import numpy as np
from skimage.transform import resize

#Extracting frames of the video
def extract_frames(video_path, n):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None, None

    cap.set(cv2.CAP_PROP_POS_FRAMES, n) 

    # Read nth frame
    ret1, frame_n = cap.read()
    if not ret1:
        print(f"Error: Could not read frame {n}.")
        cap.release()
        return None, None

    # Read (n+1)th frame
    ret2, frame_n1 = cap.read()
    if not ret2:
        print(f"Error: Could not read frame {n+1}.")
        cap.release()
        return frame_n, None  # Return nth frame and None if (n+1)th frame doesn't exist

    cap.release()
    return frame_n, frame_n1

#Landmark Detection

#Load the landmark detector
predictor_path = "/content/shape_predictor_68_face_landmarks.dat" 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

#detect the landmarks and return them as a numpy array
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    points = np.array([(p.x, p.y) for p in landmarks.parts()], dtype=np.int32)
    return points

#create a mask from the landmarks
def create_mask(image, landmarks):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(landmarks), 255)
    return mask

#extract the face using the mask
def extract_face(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

#creating a self shifted blended image by extracting face from the nth frame and blending it onto the n+1th frame
def self_shift_blended_image(image_n, image_n1, mask_n):
    h, w = image_n.shape[:2]
    # TODO - fix face_region_n extraction
    face_region_n = cv2.bitwise_and(image_n, image_n, mask=mask_n)

    # Resize extracted face to match n+1 frame dimensions
    resized_face = resize(face_region_n, (h, w), anti_aliasing=True)
    resized_face = (resized_face * 255).astype(np.uint8)

    # Blend extracted face onto frame n+1
    background = cv2.bitwise_and(image_n1, image_n1, mask=cv2.bitwise_not(mask_n))
    blended = cv2.addWeighted(resized_face, 0.7, background, 0.3, 0)

    return blended

## getter function for ssbi forgery augmentation ##
def get_self_shift_blended_image(image_path_n1,image_path_n2):
    frame_n1 = cv2.imread(image_path_n1)
    frame_n2 = cv2.imread(image_path_n2)
    if frame_n1 is None or frame_n2 is None:
        raise Exception("Error: Could not load one or both frames.")
    landmarks_n1 = get_landmarks(frame_n1)
    if landmarks_n1:
        try:
            mask_n = create_mask(frame_n1, landmarks_n1)
            ssbi_image = self_shift_blended_image(frame_n1, frame_n2, mask_n)
            return ssbi_image
        except Exception as e:
            raise Exception(f"Error during self-shift blending: {e}")
