import cv2

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
