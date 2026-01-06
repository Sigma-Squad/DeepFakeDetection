import cv2
import random
from helpers import extract_all_frames, get_landmarks, warp_face_delaunay

def create_deepfake_video(video_path, output_path=None, blending_interval=5, window_size=10, max_frames=None):
    result = extract_all_frames(video_path, max_frames)
    if result is None:
        return False

    all_frames, fps = result
    if not all_frames:
        return False

    modified_frames = all_frames.copy()
    total_frames = len(all_frames)
    blend_indices = list(range(window_size, total_frames - window_size, blending_interval))

    for i in blend_indices:
        current_frame = all_frames[i]
        landmarks_current = get_landmarks(current_frame)
        if landmarks_current is None:
            continue

        nearby_indices = [j for j in range(max(0, i - window_size), min(i + window_size + 1, total_frames))
                          if j != i and j not in blend_indices]
        if not nearby_indices:
            continue

        random_index = random.choice(nearby_indices)
        source_frame = all_frames[random_index]
        landmarks_source = get_landmarks(source_frame)
        if landmarks_source is None:
            continue

        try:
            blended = warp_face_delaunay(source_frame, current_frame, landmarks_source, landmarks_current)
            modified_frames[i] = blended
        except:
            continue

    if output_path and modified_frames:
        h, w = modified_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in modified_frames:
            out.write(frame)
        out.release()

    return True
