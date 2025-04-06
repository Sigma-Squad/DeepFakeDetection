import cv2
import numpy as np

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
