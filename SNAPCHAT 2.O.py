import cv2
import mediapipe as mp
import os
import time
import pygame
import numpy as np


# Initialize Pygame for sound
pygame.init()
beep = pygame.mixer.Sound(r"C:\second semester\Linear Algebra\sound effects\Beep.wav")
click = pygame.mixer.Sound(r"C:\second semester\Linear Algebra\sound effects\iphone-camera-capture-6448.wav")

# Webcam and MediaPipe setup
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=1)

# Load filter images
filters = {
    "cat": {
        "left_ear": cv2.imread(r"C:\second semester\Linear Algebra\LA final project\left ear.png", cv2.IMREAD_UNCHANGED),
        "right_ear": cv2.imread(r"C:\second semester\Linear Algebra\LA final project\right.png", cv2.IMREAD_UNCHANGED),
        "nose": cv2.imread(r"C:\second semester\Linear Algebra\LA final project\Mouth.png", cv2.IMREAD_UNCHANGED)
    },
    "dog": {
        "left_ear": cv2.imread(r"C:\second semester\Linear Algebra\LA final project\Dogleft.png", cv2.IMREAD_UNCHANGED),
        "right_ear": cv2.imread(r"C:\second semester\Linear Algebra\LA final project\Dogright.png", cv2.IMREAD_UNCHANGED),
        "nose": cv2.imread(r"C:\second semester\Linear Algebra\LA final project\Dognose.png", cv2.IMREAD_UNCHANGED),
        "tongue": cv2.imread(r"C:\second semester\Linear Algebra\LA final project\tongue.png", cv2.IMREAD_UNCHANGED)

    },
    "glasses": {
        "glass": cv2.imread(r"C:\second semester\Linear Algebra\LA final project\Glasses.png", cv2.IMREAD_UNCHANGED)
    },
    "vintage": {},
    "brighten": {},
    "Black & white":{},
    "cartoon":{},
    "beauty":{},
    "sketch":{},
    "negative":{},
    "sepia":{},
    "glitch":{},
    "neon_glow":{},
    "Original":{}
}

filter_names = list(filters.keys())
active_filter = filter_names[0]
current_filter = active_filter

# Create folder for pictures
os.makedirs(r"C:\second semester\Linear Algebra\LA final project\Pictures", exist_ok=True)
pic_count = 0

# Overlay helper
def overlay_image(roi, part_img, part_mask):
    if roi.shape[0] == part_img.shape[0] and roi.shape[1] == part_img.shape[1]:
        mask_inv = cv2.bitwise_not(part_mask)
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(part_img, part_img, mask=part_mask)
        return cv2.add(bg, fg)
    return roi

# Save image
def save_picture(frame):
    global pic_count
    filename = os.path.join(r"C:\second semester\Linear Algebra\LA final project\Pictures", f"{active_filter}_filter_{pic_count}.png")
    cv2.imwrite(filename, frame)
    print(f"[✅] Picture saved as {filename}")
    click.play()
    pic_count += 1

# Draw floating filter buttons
def draw_filter_buttons(frame):
    h, w, _ = frame.shape
    button_size = 20  # decreased from 60
    padding = 5      # slightly reduced padding
    y = h - button_size - 10
    for i, key in enumerate(filter_names):
        x = padding + i * (button_size + padding)
        color = (200, 200, 200) if key != active_filter else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + button_size, y + button_size), color, -1)
        cv2.putText(frame, key.upper(), (x + 3, y + 17),  # adjusted Y for smaller text
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)  # smaller text size and thickness


# Check if a button is clicked
def check_button_click(x, y, frame_height):
    button_size = 20
    padding = 5
    y_start = frame_height - button_size - 10
    for i, key in enumerate(filter_names):
        x_start = padding + i * (button_size + padding)
        if x_start <= x <= x_start + button_size and y_start <= y <= y_start + button_size:
            return key
    return None

# Mouse click handler
def mouse_callback(event, x, y, flags, param):
    global active_filter, current_filter
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        clicked = check_button_click(x, y, frame_height)
        if clicked:
            active_filter = clicked
            current_filter = clicked

cv2.namedWindow("Face Filter App 😺🐶🤓")
cv2.setMouseCallback("Face Filter App 😺🐶🤓", mouse_callback)

# Timer setup
timer_started = False
countdown_start = 0
cooldown = False
cooldown_start = 0
COOLDOWN_TIME = 5
COUNTDOWN_TIME = 3
last_beep_time = -1

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    ih, iw, _ = frame.shape
    hand_is_raised = False

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            wrist_y = hand_landmarks.landmark[0].y * ih
            index_y = hand_landmarks.landmark[8].y * ih
            if index_y < wrist_y:
                hand_is_raised = True

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]
            forehead = face_landmarks.landmark[10]

            left_eye_x, left_eye_y = int(left_eye.x * iw), int(left_eye.y * ih)
            right_eye_x, right_eye_y = int(right_eye.x * iw), int(right_eye.y * ih)
            nose_x, nose_y = int(nose_tip.x * iw), int(nose_tip.y * ih)
            forehead_x, forehead_y = int(forehead.x * iw), int(forehead.y * ih)

            ear_width = int((right_eye_x - left_eye_x) * 1.8)
            ear_height = int(ear_width * 0.75)

            filter_data = filters[current_filter]
            if current_filter == "vintage":
                sepia = cv2.transform(frame, 
                    np.array([[0.272, 0.534, 0.131],
                            [0.349, 0.686, 0.168],
                            [0.393, 0.769, 0.189]]))
                frame = np.clip(sepia, 0, 255).astype(np.uint8)

            
            if current_filter == "glasses":
                glass = filter_data["glass"]
                glass_width = int((right_eye_x - left_eye_x) * 2.0)
                glass_height = int(glass_width * 0.4)
                glass_resized = cv2.resize(glass, (glass_width, glass_height))
                glass_rgb = glass_resized[:, :, :3]
                glass_mask = glass_resized[:, :, 3]
                gx = left_eye_x - int(glass_width * 0.25)
                gy = left_eye_y - int(glass_height / 2)
                if 0 <= gx < iw - glass_width and 0 <= gy < ih - glass_height:
                    roi = frame[gy:gy + glass_height, gx:gx + glass_width]
                    frame[gy:gy + glass_height, gx:gx + glass_width] = overlay_image(roi, glass_rgb, glass_mask)
            
            elif current_filter == "neon_glow":
                # Convert to grayscale and apply a blur
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                # Apply a threshold to find edges
                _, edges = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
                
                # Convert the edges to color and blend
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                
                # Apply a neon glow effect by blending with original image
                neon_glow = cv2.addWeighted(frame, 0.7, edges, 0.3, 0)
                
                frame = neon_glow
            

            elif current_filter == "Black & white":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # to keep 3 channels
            elif current_filter == "cartoon":
                # Apply bilateral filter multiple times for stronger smoothing
                color = frame
                for _ in range(3):
                    color = cv2.bilateralFilter(color, d=9, sigmaColor=10, sigmaSpace=40)
                
                # Convert to grayscale and apply adaptive thresholding to detect edges
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 7)
                
                # Sharpen the edges by using adaptive threshold with lower contrast (C=2)
                edges = cv2.adaptiveThreshold(gray, 255,
                                            cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, blockSize=7, C=2)
                
                # Convert edges to color to make it compatible with the color image
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                
                # Combine the color image and edges to create a cartoon effect
                cartoon = cv2.bitwise_and(color, edges)
                
                # Further sharpen the edges for stronger effect (use filter2D with sharpen kernel)
                kernel = np.array([[-1, -1, -1], [-1, 30, -1], [-1, -1, -1]])  # Sharpening kernel
                cartoon = cv2.filter2D(cartoon, -1, kernel)
                
                frame = cartoon

            elif current_filter == "beauty":
                # Gentle skin smoothing
                blur = cv2.bilateralFilter(frame, d=10, sigmaColor=5, sigmaSpace=20)
                
                # Convert to HSV for brightness boost
                hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                v = cv2.add(v, 20)  # light brighten
                s = cv2.add(s, 20)  # slight saturation for glow

                final_hsv = cv2.merge((h, s, v))
                frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

                # Add a soft contrast
                frame = cv2.addWeighted(frame, 1.1, frame, 0, 5)

            elif current_filter == "negative":
                # Invert the colors of the frame
                frame = cv2.bitwise_not(frame)
            elif current_filter == "sepia":
                # Create a sepia filter matrix
                sepia_filter = np.array([[0.393, 0.769, 0.189],
                                        [0.349, 0.686, 0.168],
                                        [0.272, 0.534, 0.131]])
                
                # Apply the filter to each frame
                frame = cv2.transform(frame, sepia_filter)
                
                # Clip values to ensure they are within the valid range
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            elif current_filter == "brighten":
                # Skin smoothing and brighten (using bilateral filter)
                frame = cv2.bilateralFilter(frame, d=2, sigmaColor=20, sigmaSpace=20)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                v = cv2.add(v, 30)  # brighten
                final_hsv = cv2.merge((h, s, v))
                frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            elif current_filter == "glitch":
                # Randomly shift parts of the image horizontally
                height, width, _ = frame.shape
                glitch_frame = frame.copy()
                
                # Randomly create "glitchy" vertical strips
                for i in range(0, width, np.random.randint(20, 100)):
                    y_offset = np.random.randint(-10, 10)
                    glitch_frame[:, i:i+20] = np.roll(glitch_frame[:, i:i+20], y_offset, axis=0)
                
                frame = glitch_frame

            elif current_filter == "sketch":
                # Convert to grayscale and invert it
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                inverted_gray = cv2.bitwise_not(gray)
                
                # Blur the inverted image
                blurred = cv2.GaussianBlur(inverted_gray, (111, 111), 0)
                
                # Invert the blurred image
                inverted_blurred = cv2.bitwise_not(blurred)
                
                # Create the pencil sketch
                sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
                
                # Show the result
                frame = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

            elif current_filter == "none" or current_filter == "original":
                 pass  # Just show the raw frame
            elif current_filter in ["cat", "dog"]:
    # 🐶 If the filter is 'dog', show tongue
                if current_filter == "dog":
    # Detect if mouth is open
                    top_lip = face_landmarks.landmark[13]
                    bottom_lip = face_landmarks.landmark[14]

                    top_lip_y = int(top_lip.y * ih)
                    bottom_lip_y = int(bottom_lip.y * ih)
                    mouth_open = (bottom_lip_y - top_lip_y) > 15  # Adjust threshold if needed

                    if mouth_open:
                        tongue_img = filter_data["tongue"]
                        tongue_width = int(ear_width * 0.6)
                        tongue_height = int(tongue_width * 0.6)

                        # Position it just below the nose
                        tx = int((left_eye_x + right_eye_x) / 2) - tongue_width // 2
                        ty = nose_y + int(nose_height * 0.6)

                        tongue_resized = cv2.resize(tongue_img, (tongue_width, tongue_height))
                        tongue_rgb = tongue_resized[:, :, :3]
                        tongue_mask = tongue_resized[:, :, 3]

                        if 0 <= tx < iw - tongue_width and 0 <= ty < ih - tongue_height:
                            roi = frame[ty:ty + tongue_height, tx:tx + tongue_width]
                            frame[ty:ty + tongue_height, tx:tx + tongue_width] = overlay_image(roi, tongue_rgb, tongue_mask)

                
                # 🐱🐶 Shared rendering logic (ears + nose)
                #elif current_filter in ["cat", "dog"]:
                for part, x_offset in [("left_ear", -ear_width), ("right_ear", 0)]:
                        part_img = filter_data[part]
                        resized = cv2.resize(part_img, (ear_width, ear_height))
                        rgb = resized[:, :, :3]
                        mask = resized[:, :, 3]
                        x = forehead_x + x_offset
                        y = forehead_y - int(ear_height * 0.9)
                        if 0 <= x < iw - ear_width and 0 <= y < ih - ear_height:
                            roi = frame[y:y + ear_height, x:x + ear_width]
                            frame[y:y + ear_height, x:x + ear_width] = overlay_image(roi, rgb, mask)

                            nose_width = int(ear_width * 0.7)
                            nose_height = int(nose_width * 0.5)
                            nose_img = filter_data["nose"]
                            nose_resized = cv2.resize(nose_img, (nose_width, nose_height))
                            nose_rgb = nose_resized[:, :, :3]
                            nose_mask = nose_resized[:, :, 3]
                            nx = int((left_eye_x + right_eye_x) / 2) - nose_width // 2
                            ny = nose_y - nose_height // 2
                            if 0 <= nx < iw - nose_width and 0 <= ny < ih - nose_height:
                                roi = frame[ny:ny + nose_height, nx:nx + nose_width]
                                frame[ny:ny + nose_height, nx:nx + nose_width] = overlay_image(roi, nose_rgb, nose_mask)
            

    # Handle timer for photo capture
    current_time = time.time()
    if hand_is_raised and not timer_started and not cooldown:
        timer_started = True
        countdown_start = current_time
        last_beep_time = -1

    if timer_started:
        elapsed = int(current_time - countdown_start)
        time_left = COUNTDOWN_TIME - elapsed
        if time_left > 0:
            cv2.putText(frame, str(time_left), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            if elapsed != last_beep_time:
                if not pygame.mixer.get_busy():
                    beep.play()
                last_beep_time = elapsed
        else:
            save_picture(frame)
            timer_started = False
            cooldown = False
            cooldown_start = current_time

    #if cooldown and (current_time - cooldown_start > COOLDOWN_TIME):
      #  cooldown = False

    draw_filter_buttons(frame)
    cv2.imshow("Face Filter App 😺🐶🤓", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        save_picture(frame)

cap.release()
cv2.destroyAllWindows()
