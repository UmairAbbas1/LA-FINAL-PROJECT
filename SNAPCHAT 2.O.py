import cv2
import mediapipe as mp
import os
import time
import numpy as np

# -------------------------
# Setup: Create folder for saved pictures
# -------------------------
SAVE_DIR = r"D:\LA FINAL PROJECT\Pictures"
os.makedirs(SAVE_DIR, exist_ok=True)
pic_count = 0

def save_picture(frame_to_save):
    global pic_count
    # Save the picture without UI overlays.
    filename = os.path.join(SAVE_DIR, f"{active_filter}_filter_{pic_count}.png")
    cv2.imwrite(filename, frame_to_save)
    print(f"[✅] Picture saved as {filename}")
    pic_count += 1

# -------------------------
# Countdown and cooldown settings (for auto-saving via hand raise)
# -------------------------
COUNTDOWN_TIME = 3  # seconds countdown before saving
COOLDOWN_TIME = 5   # seconds delay after auto-save
timer_started = False
countdown_start = 0
cooldown = False
cooldown_start = 0

# -------------------------
# Initialize webcam and MediaPipe modules
# -------------------------
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# -------------------------
# Load filter images
# -------------------------
# Cat filter images
cat_filter = {
    "left_ear": cv2.imread(r"D:\LA FINAL PROJECT\left.png", cv2.IMREAD_UNCHANGED),
    "right_ear": cv2.imread(r"D:\LA FINAL PROJECT\right.png", cv2.IMREAD_UNCHANGED),
    "nose": cv2.imread(r"D:\LA FINAL PROJECT\nose.png", cv2.IMREAD_UNCHANGED)
}

# Dog filter images (including tongue)
dog_filter = {
    "left_ear": cv2.imread(r"D:\LA FINAL PROJECT\Dogleft.png", cv2.IMREAD_UNCHANGED),
    "right_ear": cv2.imread(r"D:\LA FINAL PROJECT\Dogright.png", cv2.IMREAD_UNCHANGED),
    "nose": cv2.imread(r"D:\LA FINAL PROJECT\Dognose.png", cv2.IMREAD_UNCHANGED),
    "tongue": cv2.imread(r"D:\LA FINAL PROJECT\tongue.png", cv2.IMREAD_UNCHANGED)
}

# Glasses filter image (glasses overlay only)
glasses_filter = {
    "glasses": cv2.imread(r"D:\LA FINAL PROJECT\Glasses.png", cv2.IMREAD_UNCHANGED)
}

# List of available filters
filter_list = ["cat", "dog", "glasses"]

# Dictionary mapping filter name to its image data
filters = {
    "cat": cat_filter,
    "dog": dog_filter,
    "glasses": glasses_filter
}

# Default active filter
active_filter = "cat"

# -------------------------
# Helper function: Overlay an image (with alpha channel) onto an ROI
# -------------------------
def overlay_image(roi, part_img, part_mask):
    if roi.shape[0] == part_img.shape[0] and roi.shape[1] == part_img.shape[1]:
        mask_inv = cv2.bitwise_not(part_mask)
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(part_img, part_img, mask=part_mask)
        return cv2.add(bg, fg)
    return roi

# -------------------------
# Draw floating filter buttons on frame (for mouse control)
# -------------------------
def draw_filter_buttons(frame):
    h, w, _ = frame.shape
    button_size = 60
    padding = 20
    y = h - button_size - 10
    for i, key in enumerate(filter_list):
        x = padding + i * (button_size + padding)
        # Draw rectangle button; highlight if active
        color = (200, 200, 200) if key != active_filter else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + button_size, y + button_size), color, -1)
        cv2.putText(frame, key.upper(), (x + 5, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# -------------------------
# Check if a filter button is clicked based on x, y coordinates
# -------------------------
def check_button_click(x, y, frame_height):
    button_size = 60
    padding = 20
    y_start = frame_height - button_size - 10
    for i, key in enumerate(filter_list):
        x_start = padding + i * (button_size + padding)
        if x_start <= x <= x_start + button_size and y_start <= y <= y_start + button_size:
            return key
    return None

# -------------------------
# Mouse callback for filter selection
# -------------------------
def mouse_callback(event, x, y, flags, param):
    global active_filter
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        clicked = check_button_click(x, y, frame_height)
        if clicked:
            active_filter = clicked

cv2.namedWindow("SnapFilter")
cv2.setMouseCallback("SnapFilter", mouse_callback)

# -------------------------
# Main Loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)
    ih, iw, _ = frame.shape

    # Create a copy to hold overlays for saving (clean_frame)
    clean_frame = frame.copy()

    # -------------------------
    # Hand detection: trigger auto-save when hand is raised
    # -------------------------
    hand_is_raised = False
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            wrist_y = hand_landmarks.landmark[0].y * ih
            index_y = hand_landmarks.landmark[8].y * ih
            if index_y < wrist_y:  # simple check for raised hand
                hand_is_raised = True

    # -------------------------
    # Process face landmarks to apply filters (draw overlays on clean_frame)
    # -------------------------
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Extract key landmarks:
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]
            forehead = face_landmarks.landmark[10]
            # For dog's tongue, use lip landmarks 13 (upper) and 14 (lower)
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            
            # Compute pixel coordinates
            left_eye_x, left_eye_y = int(left_eye.x * iw), int(left_eye.y * ih)
            right_eye_x, right_eye_y = int(right_eye.x * iw), int(right_eye.y * ih)
            nose_x, nose_y = int(nose_tip.x * iw), int(nose_tip.y * ih)
            forehead_x, forehead_y = int(forehead.x * iw), int(forehead.y * ih)
            mouth_open_gap = (lower_lip.y - upper_lip.y) * ih  # gap in pixels
            
            if active_filter in ["cat", "dog"]:
                filter_data = filters[active_filter]
                left_ear_img = filter_data["left_ear"]
                right_ear_img = filter_data["right_ear"]
                nose_img = filter_data["nose"]
                
                ear_width = int((right_eye_x - left_eye_x) * 1.8)
                ear_height = int(ear_width * 0.75)
                
                # Position ears relative to the forehead:
                left_ear_x = forehead_x - ear_width
                left_ear_y = forehead_y - int(ear_height * 0.9)
                right_ear_x = forehead_x
                right_ear_y = forehead_y - int(ear_height * 0.9)
                
                left_ear_resized = cv2.resize(left_ear_img, (ear_width, ear_height))
                left_ear_rgb = left_ear_resized[:, :, :3]
                left_ear_mask = left_ear_resized[:, :, 3]
                
                right_ear_resized = cv2.resize(right_ear_img, (ear_width, ear_height))
                right_ear_rgb = right_ear_resized[:, :, :3]
                right_ear_mask = right_ear_resized[:, :, 3]
                
                for (img_rgb, mask, x, y) in [
                    (left_ear_rgb, left_ear_mask, left_ear_x, left_ear_y),
                    (right_ear_rgb, right_ear_mask, right_ear_x, right_ear_y)
                ]:
                    if 0 <= x < iw - ear_width and 0 <= y < ih - ear_height:
                        roi = clean_frame[y:y + ear_height, x:x + ear_width]
                        clean_frame[y:y + ear_height, x:x + ear_width] = overlay_image(roi, img_rgb, mask)
                
                nose_width = int(ear_width * 0.45)
                nose_height = int(nose_width * 0.6)
                nx = int((left_eye_x + right_eye_x) / 2) - nose_width // 2
                ny = nose_y - nose_height // 2
                
                nose_resized = cv2.resize(nose_img, (nose_width, nose_height))
                nose_rgb = nose_resized[:, :, :3]
                nose_mask = nose_resized[:, :, 3]
                if 0 <= nx < iw - nose_width and 0 <= ny < ih - nose_height:
                    roi_nose = clean_frame[ny:ny + nose_height, nx:nx + nose_width]
                    clean_frame[ny:ny + nose_height, nx:nx + nose_width] = overlay_image(roi_nose, nose_rgb, nose_mask)
                
                # For dog filter: add tongue overlay when mouth is open
                if active_filter == "dog":
                    mouth_threshold = ear_height * 0.3  # adjust threshold as needed
                    if mouth_open_gap > mouth_threshold and "tongue" in filter_data:
                        tongue_img = filter_data["tongue"]
                        tongue_width = int(nose_width * 1.2)
                        tongue_height = int(tongue_width * 0.6)
                        tx = nx + nose_width // 2 - tongue_width // 2
                        ty = ny + nose_height  # position just below nose
                        tongue_resized = cv2.resize(tongue_img, (tongue_width, tongue_height))
                        tongue_rgb = tongue_resized[:, :, :3]
                        tongue_mask = tongue_resized[:, :, 3]
                        if 0 <= tx < iw - tongue_width and 0 <= ty < ih - tongue_height:
                            roi_tongue = clean_frame[ty:ty + tongue_height, tx:tx + tongue_width]
                            clean_frame[ty:ty + tongue_height, tx:tx + tongue_width] = overlay_image(roi_tongue, tongue_rgb, tongue_mask)
            
            elif active_filter == "glasses":
                # Glasses overlay based on eye landmarks
                glasses_img = filters["glasses"]["glasses"]
                glasses_width = int((right_eye_x - left_eye_x) * 2.0)
                glasses_height = int(glasses_width * 0.4)+50
                gx = left_eye_x - int(0.3 * glasses_width)
                gy = left_eye_y - int(glasses_height * 1.2)+90
                glasses_resized = cv2.resize(glasses_img, (glasses_width, glasses_height))
                glasses_rgb = glasses_resized[:, :, :3]
                glasses_mask = glasses_resized[:, :, 3]
                if 0 <= gx < iw - glasses_width and 0 <= gy < ih - glasses_height:
                    roi_glasses = clean_frame[gy:gy + glasses_height, gx:gx + glasses_width]
                    clean_frame[gy:gy + glasses_height, gx:gx + glasses_width] = overlay_image(roi_glasses, glasses_rgb, glasses_mask)
    
    # -------------------------
    # Countdown & Cooldown logic for auto-save (triggered by hand raise)
    # -------------------------
    current_time = time.time()
    if hand_is_raised and not timer_started and not cooldown:
        timer_started = True
        countdown_start = current_time

    if timer_started:
        elapsed = int(current_time - countdown_start)
        if elapsed < COUNTDOWN_TIME:
            # Draw countdown on the displayed frame (but not on clean_frame)
            cv2.putText(frame, str(COUNTDOWN_TIME - elapsed), (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        else:
            save_picture(clean_frame)
            timer_started = False
            cooldown = True
            cooldown_start = current_time

    if cooldown and (current_time - cooldown_start > COOLDOWN_TIME):
        cooldown = False

    # -------------------------
    # Compose the display frame with UI overlays (without affecting clean_frame)
    # -------------------------
    display_frame = clean_frame.copy()
    draw_filter_buttons(display_frame)
    cv2.putText(display_frame, f"Active Filter: {active_filter.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # -------------------------
    # Handle key presses: 'q' to quit, 's' to save manually.
    # (Filter switching is now done with mouse.)
    # -------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_picture(clean_frame)
    
    cv2.imshow("SnapFilter", display_frame)

cap.release()
cv2.destroyAllWindows()
