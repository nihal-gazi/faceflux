# --- Protobuf Compatibility Fix for Protobuf >= 4.26 / 5.x ---
# Must execute before importing mediapipe to prevent 'MessageFactory' object has no attribute 'GetPrototype'
try:
    import google.protobuf.message_factory as mf
    if hasattr(mf, "MessageFactory") and not hasattr(mf.MessageFactory, "GetPrototype"):
        mf.MessageFactory.GetPrototype = lambda self, descriptor: mf.GetMessageClass(descriptor)
except Exception:
    pass

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

# --- MediaPipe Initialization (relative model path) ---
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading MediaPipe Face Landmarker model...")
    try:
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading MediaPipe model: {e}")

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    face_mesh = vision.FaceLandmarker.create_from_options(options)
    print("MediaPipe Face Landmarker initialized successfully.")
except Exception as e:
    print(f"Error: Could not initialize MediaPipe Face Landmarker. Reason: {e}")
    face_mesh = None


# --- Helper Functions ---

def get_face_mask_box(img, feather_pct, padding_pct):
    """
    Detects face mesh in the image and returns a feathered alpha mask of the face
    along with its padded bounding box (x, y, w, h).
    """
    if face_mesh is None:
        return None, None
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    results = face_mesh.detect(mp_image)
    if not results.face_landmarks:
        return None, None

    pts = np.array([(int(p.x * w), int(p.y * h)) for p in results.face_landmarks[0]], np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    x, y, bw, bh = cv2.boundingRect(hull)
    pad = int(max(bw, bh) * padding_pct)
    x_pad = max(x - pad, 0)
    y_pad = max(y - pad, 0)
    x2 = min(x + bw + pad, w)
    y2 = min(y + bh + pad, h)
    mask_roi = mask[y_pad:y2, x_pad:x2]
    if feather_pct > 0 and mask_roi.size > 0:
        k = int(min(mask_roi.shape[0], mask_roi.shape[1]) * feather_pct)
        if k % 2 == 0:
            k += 1
        mask_roi = cv2.GaussianBlur(mask_roi, (k, k), 0)
    return mask_roi, (x_pad, y_pad, x2 - x_pad, y2 - y_pad)


def get_landmarks(img, landmark_step=1):
    """
    Extracts face landmark points plus frame corner points for Delaunay triangulation.
    """
    if img is None or face_mesh is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = face_mesh.detect(mp_image)
    except Exception:
        return None
    if not results.face_landmarks:
        return None
    landmarks_mp = results.face_landmarks[0]
    h, w, _ = img.shape
    pts = np.array([(pt.x * w, pt.y * h) for pt in landmarks_mp], dtype=np.float32)
    landmarks = pts[::landmark_step] if landmark_step > 1 else pts
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float32)
    return np.vstack((landmarks, corners))


def calculate_delaunay_triangles(rect, points):
    """
    Calculates Delaunay triangulation vertex indices for given points within bounding rectangle.
    """
    if points is None or len(points) < 3:
        return []
    points[:, 0] = np.clip(points[:, 0], rect[0], rect[0] + rect[2] - 1)
    points[:, 1] = np.clip(points[:, 1], rect[1], rect[1] + rect[3] - 1)
    subdiv = cv2.Subdiv2D(rect)
    inserted = {}
    for i, p in enumerate(points):
        key = (int(p[0]), int(p[1]))
        if key not in inserted:
            try:
                subdiv.insert(key)
                inserted[key] = i
            except cv2.error:
                continue
    tris = subdiv.getTriangleList()
    delaunay = []
    for t in tris:
        coords = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]
        if all(rect[0] <= x < rect[0] + rect[2] and rect[1] <= y < rect[1] + rect[3] for x, y in coords):
            idxs = [inserted.get(c) for c in coords]
            if all(i is not None for i in idxs) and len(set(idxs)) == 3:
                delaunay.append(idxs)
    return delaunay


def warp_triangle(img1, img2, t1, t2):
    """
    Warps and copies a triangular region from img1 (at vertices t1) to img2 (at vertices t2).
    """
    if len(t1) != 3 or len(t2) != 3:
        return
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return
    src = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if src.size == 0:
        return
    t1r = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2r = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2r), (1, 1, 1), 16)
    M = cv2.getAffineTransform(np.float32(t1r), np.float32(t2r))
    warped = cv2.warpAffine(src, M, (r2[2], r2[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warped = warped.astype(np.float32) * mask
    y1, y2 = r2[1], r2[1] + r2[3]
    x1, x2 = r2[0], r2[0] + r2[2]
    img2[y1:y2, x1:x2] = img2[y1:y2, x1:x2] * (1 - mask) + warped


def morph_faces(img1, img2, alpha, dim, step):
    """
    Morphs img1 and img2 based on alpha interpolation (0: img1, 1: img2) at resolution dim x dim.
    """
    if img1 is None or img2 is None:
        return np.zeros((dim, dim, 3), dtype=np.uint8)
    a = cv2.resize(img1, (dim, dim)).astype(np.float32) / 255.0
    b = cv2.resize(img2, (dim, dim)).astype(np.float32) / 255.0
    l1 = get_landmarks((a * 255).astype(np.uint8), step)
    l2 = get_landmarks((b * 255).astype(np.uint8), step)
    if l1 is None or l2 is None or l1.shape != l2.shape:
        return cv2.addWeighted((a * 255).astype(np.uint8), 1 - alpha, (b * 255).astype(np.uint8), alpha, 0)
    m = (1 - alpha) * l1 + alpha * l2
    tris = calculate_delaunay_triangles((0, 0, dim, dim), m)
    if not tris:
        return cv2.addWeighted((a * 255).astype(np.uint8), 1 - alpha, (b * 255).astype(np.uint8), alpha, 0)
    Wa = np.zeros_like(a)
    Wb = np.zeros_like(b)
    for ids in tris:
        warp_triangle(a, Wa, l1[ids], m[ids])
        warp_triangle(b, Wb, l2[ids], m[ids])
    out = (1 - alpha) * Wa + alpha * Wb
    return (out * 255).astype(np.uint8)
