import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load FASIAL.png
BASE_DIR = r"D:\LA-FINAL-PROJECT"
FASIAL_PATH = os.path.join(BASE_DIR, "testingf2.png")
fasial_img = cv2.imread(FASIAL_PATH, cv2.IMREAD_UNCHANGED)
if fasial_img is None:
    print("Error: FASIAL.png not found or could not be loaded.")
    exit()
alpha_fasial = fasial_img[:, :, 3]
mask_fasial = (alpha_fasial > 200).astype(np.uint8) * 255
cnts, _ = cv2.findContours(mask_fasial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if cnts:
    x, y, w, h = cv2.boundingRect(cnts[0])
    fasial_rgb = fasial_img[y:y+h, x:x+w, :3]
    fasial_mask = mask_fasial[y:y+h, x:x+w]
else:
    print("Error: No contours found in FASIAL.png mask.")
    exit()

# Initialize webcam with a lower resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

# Precompute the dark theme background effect
def create_dark_theme_background(iw, ih):
    # Create a dark charcoal to black gradient
    gradient = np.zeros((ih, iw, 3), dtype=np.uint8)
    for i in range(ih):
        t = i / ih
        color = (int(30 * (1 - t) + 10 * t), int(30 * (1 - t) + 10 * t), int(40 * (1 - t) + 20 * t))  # Dark charcoal to black
        gradient[i, :] = color

    # Add subtle silver gradient overlay
    silver_gradient = np.zeros((ih, iw, 3), dtype=np.uint8)
    for i in range(ih):
        t = i / ih
        silver = int(50 * (1 - t))  # Fades from silver to transparent
        silver_gradient[i, :] = (silver, silver, silver)

    # Add starry effect (small white dots)
    stars = np.zeros((ih, iw, 3), dtype=np.uint8)
    num_stars = 50
    for _ in range(num_stars):
        x = np.random.randint(0, iw)
        y = np.random.randint(0, ih)
        brightness = np.random.randint(100, 255)
        cv2.circle(stars, (x, y), 1, (brightness, brightness, brightness), -1)

    return gradient, silver_gradient, stars

# Precompute background for the webcam resolution
ret, frame = cap.read()
if not ret:
    print("Error: Failed to capture initial frame for resolution.")
    exit()
ih, iw, _ = frame.shape
gradient, silver_gradient, stars = create_dark_theme_background(iw, ih)

def overlay_image(roi, img, m):
    img = img.astype(np.uint8)
    m = m.astype(np.uint8)
    if roi.shape[:2] != img.shape[:2]:
        img = cv2.resize(img, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)
        m = cv2.resize(m, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    inv = cv2.bitwise_not(m)
    bg = cv2.bitwise_and(roi, roi, mask=inv)
    fg = cv2.bitwise_and(img, img, mask=m)
    return cv2.add(bg, fg)

def apply_dark_theme_background(frame, gradient, silver_gradient, stars):
    # Apply dark gradient
    frame[:] = cv2.addWeighted(frame, 0.6, gradient, 0.4, 0)
    # Add silver gradient
    frame[:] = cv2.addWeighted(frame, 0.8, silver_gradient, 0.2, 0)
    # Add stars
    frame[:] = cv2.addWeighted(frame, 0.9, stars, 0.1, 0)
    return frame

def get_face_box(landmarks, w, h, vertical_offset=30):
    if not landmarks:
        return 0, 0, w, h
    pts = np.array([[p.x*w, p.y*h] for p in landmarks.landmark])
    hull_idx = [10, 152, 234, 454]
    hull = pts[hull_idx]
    x, y, bw, bh = cv2.boundingRect(hull.astype(np.float32))
    pad_w, pad_h = int(bw*0.3), int(bh*0.4)
    x0 = max(0, x-pad_w)
    y0 = max(0, y-pad_h-vertical_offset)
    return x0, y0, min(w, x+bw+pad_w)-x0, min(h, y+bh)-y0

# Variables for optimization
last_landmarks = None
frame_counter = 0
last_size = None
cached_donor = None
cached_mask = None
TARGET_FPS = 30
FRAME_DURATION = 1.0 / TARGET_FPS

# Main loop
while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    frame = cv2.flip(frame, 1)

    # Apply dark theme background effect using precomputed data
    frame = apply_dark_theme_background(frame, gradient, silver_gradient, stars)

    # Face detection optimization: run every 5 frames
    frame_counter += 1
    if frame_counter % 5 == 0 or last_landmarks is None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            last_landmarks = results.multi_face_landmarks[0]
    
    # Overlay FASIAL.png if face is detected
    if last_landmarks:
        x0, y0, bw, bh = get_face_box(last_landmarks, iw, ih)
        
        if x0 >= 0 and y0 >= 0 and bw > 0 and bh > 0:
            current_size = (bw, bh)
            if last_size != current_size or cached_donor is None or cached_mask is None:
                cached_donor = cv2.resize(fasial_rgb, (bw, bh), interpolation=cv2.INTER_AREA)
                cached_mask = cv2.resize(fasial_mask, (bw, bh), interpolation=cv2.INTER_NEAREST)
                last_size = current_size
            
            h_clip = min(bh, ih - y0)
            w_clip = min(bw, iw - x0)
            if h_clip > 0 and w_clip > 0 and cached_donor.shape[0] >= h_clip and cached_donor.shape[1] >= w_clip and cached_mask.shape[0] >= h_clip and cached_mask.shape[1] >= w_clip:
                donor = cached_donor[:h_clip, :w_clip]
                msk = cached_mask[:h_clip, :w_clip]
                roi = frame[y0:y0+h_clip, x0:x0+w_clip]
                frame[y0:y0+h_clip, x0:x0+w_clip] = overlay_image(roi, donor, msk)
    
    cv2.imshow("Fasial Filter Test - Elite Tribute", frame)
    
    # Control frame rate
    elapsed_time = time.time() - start_time
    sleep_time = max(0, FRAME_DURATION - elapsed_time)
    time.sleep(sleep_time)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()