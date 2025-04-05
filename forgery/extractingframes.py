#install the following 
#pip install dlib opencv-python numpy albumentations scikit-image


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

    cap.set(cv2.CAP_PROP_POS_FRAMES, n)  # Set video to nth frame

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

video_file = "video.mp4"  # Replace with your video file path
n = 200  # Frame index to extract
frame_n, frame_n1 = extract_frames(video_file, n)

if frame_n is not None:
    print(f"Displaying Frame {n}:")
    cv2.imshow(frame_n)

if frame_n1 is not None:
    print(f"Displaying Frame {n+1}:")
    cv2.imshow(frame_n1)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Save frames as PNG images
if frame_n is not None:
    cv2.imwrite(f"frame_{n}.png", frame_n)  # Save nth frame as PNG
    print(f"Frame {n} saved as frame_{n}.png")

if frame_n1 is not None:
    cv2.imwrite(f"frame_{n+1}.png", frame_n1)  # Save (n+1)th frame as PNG
    print(f"Frame {n+1} saved as frame_{n+1}.png")




#Landmark Detection

#Load the landmark detector
predictor_path = "/content/shape_predictor_68_face_landmarks.dat" #replace with the path of your landmark detection file
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
def apply_self_shift_blend(image_n, image_n1, mask_n):
    h, w = image_n.shape[:2]

    # Extract face region from frame_n
    face_region_n = cv2.bitwise_and(image_n, image_n, mask=mask_n)

    # Resize extracted face to match n+1 frame dimensions
    resized_face = resize(face_region_n, (h, w), anti_aliasing=True)
    resized_face = (resized_face * 255).astype(np.uint8)

    # Blend extracted face onto frame n+1
    background = cv2.bitwise_and(image_n1, image_n1, mask=cv2.bitwise_not(mask_n))
    blended = cv2.addWeighted(resized_face, 0.7, background, 0.3, 0)

    return blended




# Load frames n and n+1
frame_n_path = "/content/frame_200.png"  # nth frame
frame_n1_path = "/content/frame_201.png"  # (n+1)th frame

frame_n = cv2.imread(frame_n_path)
frame_n1 = cv2.imread(frame_n1_path)

if frame_n is None or frame_n1 is None:
    print("Error: Could not load one or both frames.")
else:
    # Get landmarks from frame the nth frame
    landmarks_n = get_landmarks(frame_n)

    if landmarks_n is not None:
        # Create mask from frame nth frame
        mask_n = create_mask(frame_n, landmarks_n)

        # Apply self-shifted blending onto frame (n+1)th frame
        ssbi_image = apply_self_shift_blend(frame_n, frame_n1, mask_n)

        # Show final SSBI output
        cv2.imshow(ssbi_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No face detected in frame 200.")
