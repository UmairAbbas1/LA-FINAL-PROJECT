

import cv2
import mediapipe as mp
import os
import time
import pygame
import numpy as np
import random
import math

################################################################################
#                           SOUND & INITIAL SETUP                               #
################################################################################

pygame.init()

# Load sounds (adjust paths if needed)
beep = pygame.mixer.Sound(r"D:\LA-FINAL-PROJECT\Beep.wav")
click = pygame.mixer.Sound(r"D:\LA-FINAL-PROJECT\iphone-camera-capture-6448.wav")
cowboy_music = pygame.mixer.Sound(r"D:\LA-FINAL-PROJECT\cowboy_music.wav")

# Webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Unable to open webcam.")
    exit(1)

# MediaPipe face mesh & hands
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(max_num_hands=1)

################################################################################
#                            FILTER LOADING & DICTS                             #
################################################################################
# We'll keep your original dictionary with cat/dog/cowboy/glasses, plus placeholders
# for color-based filters like glitch, neon_glow, etc.

filters = {
    "cat": {
        "left_ear":  cv2.imread(r"D:\LA-FINAL-PROJECT\left.png",  cv2.IMREAD_UNCHANGED),
        "right_ear": cv2.imread(r"D:\LA-FINAL-PROJECT\right.png", cv2.IMREAD_UNCHANGED),
        "nose":      cv2.imread(r"D:\LA-FINAL-PROJECT\nose.png",  cv2.IMREAD_UNCHANGED)
    },
    "dog": {
        "left_ear":  cv2.imread(r"D:\LA-FINAL-PROJECT\Dogleft.png",   cv2.IMREAD_UNCHANGED),
        "right_ear": cv2.imread(r"D:\LA-FINAL-PROJECT\Dogright.png",  cv2.IMREAD_UNCHANGED),
        "nose":      cv2.imread(r"D:\LA-FINAL-PROJECT\Dognose.png",   cv2.IMREAD_UNCHANGED),
        "tongue":    cv2.imread(r"D:\LA-FINAL-PROJECT\tongue.png",    cv2.IMREAD_UNCHANGED)
    },
    "glasses": {
        "glass": cv2.imread(r"D:\LA-FINAL-PROJECT\Glasses.png", cv2.IMREAD_UNCHANGED)
    },
    "cowboy": {
        "hat":      cv2.imread(r"D:\LA-FINAL-PROJECT\cowboy_hat.png",     cv2.IMREAD_UNCHANGED),
        "mustache": cv2.imread(r"D:\LA-FINAL-PROJECT\moustache.png",      cv2.IMREAD_UNCHANGED),
        #"glasses":  cv2.imread(r"D:\LA-FINAL-PROJECT\cowboy_glasses.png", cv2.IMREAD_UNCHANGED)
    },
    "vintage":      {},
    "brighten":     {},
    "Black & white":{},
    "cartoon":      {},
    "beauty":       {},
    "sketch":       {},
    "negative":     {},
    "sepia":        {},
    "glitch":       {},
    "neon_glow":    {},
    "Original":     {}
}

# The list of all filter names in your UI
filter_names = list(filters.keys())

# Active/Current filter states
active_filter = filter_names[0]
current_filter = active_filter
last_filter = None

# Directory to save pictures
os.makedirs(r"D:\LA-FINAL-PROJECT\Pictures", exist_ok=True)
pic_count = 0

################################################################################
#                     PANEL STATES & GEOMETRY FOR SCROLLING                    #
################################################################################

# Toggle for panel open/close
filter_panel_open = False

# Panel origin (where the top bar is anchored)
panel_x, panel_y = 10, 60

# Draggable
panel_drag = False

# Dimensions
panel_width = 160
visible_panel_height = 300
drag_bar_height = 20
arrow_zone_height = 30

button_height = 40
padding = 10

# We'll move one button's worth on each arrow click
SCROLL_STEP = button_height + padding

# Scroll offset
panel_scroll_offset = 0
MIN_SCROLL_OFFSET = 0
MAX_SCROLL_OFFSET = 0

################################################################################
#                     COUNTDOWN & COOLDOWN TIMING LOGIC                        #
################################################################################

timer_started = False
countdown_start = 0
cooldown = False
cooldown_start = 0
COOLDOWN_TIME = 5   # after taking a picture, wait 5 seconds
COUNTDOWN_TIME = 3  # 3-second countdown
last_beep_time = -1 # track beep intervals

################################################################################
#                   HELPER FUNCTIONS: OVERLAY & SAVE PICS                      #
################################################################################

def overlay_image(roi, part_img, part_mask):
    """
    Overlays a PNG with alpha channel onto ROI (roi).
    Ensures the images and mask are the same size via resizing if needed.
    """
    roi = roi.astype(np.uint8)
    part_img = part_img.astype(np.uint8)
    part_mask = part_mask.astype(np.uint8)

    if roi.shape[:2] != part_img.shape[:2] or roi.shape[:2] != part_mask.shape[:2]:
        part_img = cv2.resize(
            part_img, (roi.shape[1], roi.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        part_mask = cv2.resize(
            part_mask, (roi.shape[1], roi.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    mask_inv = cv2.bitwise_not(part_mask)
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(part_img, part_img, mask=part_mask)
    return cv2.add(bg, fg)

def save_picture(frame):
    """
    Saves the current frame with the active filter name + incremental counter.
    """
    global pic_count
    filename = os.path.join(
        r"D:\LA-FINAL-PROJECT\Pictures",
        f"{active_filter}_filter_{pic_count}.png"
    )
    cv2.imwrite(filename, frame)
    print(f"[✅] Picture saved as {filename}")
    click.play()
    pic_count += 1

################################################################################
#                      DRAW THE TOGGLE & DRAGGABLE PANEL                       #
################################################################################

def draw_toggle_button(frame):
    """
    A small rectangle in the top-left corner for toggling the filter panel open/close.
    """
    global filter_panel_open

    x, y = 10, 10
    box_w = 30
    box_h = 30

    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), -1)
    arrow = "<" if filter_panel_open else ">"
    cv2.putText(
        frame, arrow,
        (x + 7, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 0, 0), 2
    )

def draw_filter_panel(frame):
    """
    Draws a **scrollable** panel with:
     - a draggable top bar
     - an Up arrow region
     - a Down arrow region
     - a list of filter buttons scrolled by panel_scroll_offset
    """
    global filter_panel_open
    global panel_scroll_offset, MIN_SCROLL_OFFSET, MAX_SCROLL_OFFSET
    global panel_x, panel_y
    if not filter_panel_open:
        return

    # The bar above the panel for dragging:
    top_bar_y1 = panel_y - drag_bar_height
    top_bar_y2 = panel_y

    # The bounding rectangle for the entire panel:
    overall_top = top_bar_y1
    overall_bottom = panel_y + visible_panel_height

    # Outer border
    cv2.rectangle(
        frame,
        (panel_x, overall_top),
        (panel_x + panel_width, overall_bottom),
        (50, 50, 50), 2
    )

    # Draggable bar
    cv2.rectangle(
        frame,
        (panel_x, top_bar_y1),
        (panel_x + panel_width, top_bar_y2),
        (100, 100, 100), -1
    )
    cv2.putText(
        frame, "Move panel",
        (panel_x + 10, panel_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
        (255, 255, 255), 1
    )

    # Up arrow region
    up_arrow_y1 = panel_y
    up_arrow_y2 = panel_y + arrow_zone_height
    cv2.rectangle(
        frame,
        (panel_x, up_arrow_y1),
        (panel_x + panel_width, up_arrow_y2),
        (120, 120, 120), -1
    )
    cv2.putText(
        frame, "Up",
        (panel_x + 60, up_arrow_y2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 1
    )

    # Down arrow region
    down_arrow_y1 = overall_bottom - arrow_zone_height
    down_arrow_y2 = overall_bottom
    cv2.rectangle(
        frame,
        (panel_x, down_arrow_y1),
        (panel_x + panel_width, down_arrow_y2),
        (120, 120, 120), -1
    )
    cv2.putText(
        frame, "Down",
        (panel_x + 45, down_arrow_y2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 1
    )

    # The scrollable region is between up_arrow_y2 and down_arrow_y1
    scroll_area_top = up_arrow_y2
    scroll_area_bottom = down_arrow_y1
    scroll_area_height = scroll_area_bottom - scroll_area_top

    # Calculate how big the list is
    num_filters = len(filter_names)
    total_list_height = num_filters * button_height + (num_filters - 1) * padding

    # Overhang => how much bigger the list is than the visible scroll area
    overhang = total_list_height - scroll_area_height
    if overhang < 0:
        overhang = 0
    MIN_SCROLL_OFFSET = -overhang
    MAX_SCROLL_OFFSET = 0

    # Clamp
    if panel_scroll_offset < MIN_SCROLL_OFFSET:
        panel_scroll_offset = MIN_SCROLL_OFFSET
    elif panel_scroll_offset > MAX_SCROLL_OFFSET:
        panel_scroll_offset = MAX_SCROLL_OFFSET

    # Now draw each filter button with the offset
    current_y = scroll_area_top + panel_scroll_offset
    for i, key in enumerate(filter_names):
        y_pos = current_y + i * (button_height + padding)

        # If button is above the visible area, skip
        if y_pos + button_height < scroll_area_top:
            continue
        # If button is below the visible area, break
        if y_pos > scroll_area_bottom - button_height:
            break

        color = (200, 200, 200) if key != active_filter else (0, 255, 255)
        cv2.rectangle(
            frame,
            (panel_x, y_pos),
            (panel_x + panel_width, y_pos + button_height),
            color, -1
        )
        cv2.putText(
            frame, key[:12],
            (panel_x + 5, y_pos + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 0), 1
        )

################################################################################
#                           MOUSE CALLBACK FOR UI                              #
################################################################################

def mouse_callback(event, x, y, flags, param):
    """
    - Toggle the panel (arrow at top-left corner).
    - Drag the panel by the top bar.
    - Scroll up/down by arrow clicks.
    - Select a filter if click is in the scroll region.
    """
    global filter_panel_open
    global panel_drag, panel_x, panel_y
    global panel_scroll_offset, active_filter, current_filter

    if event == cv2.EVENT_LBUTTONDOWN:
        # 1) Check toggle arrow
        if 10 <= x <= 40 and 10 <= y <= 40:
            filter_panel_open = not filter_panel_open
            return

        # 2) If the panel is open, see if we clicked inside it
        if filter_panel_open:
            # Draggable bar
            top_bar_y1 = panel_y - drag_bar_height
            top_bar_y2 = panel_y

            # Up arrow region
            up_arrow_y1 = panel_y
            up_arrow_y2 = panel_y + arrow_zone_height

            # Down arrow region
            down_arrow_y1 = panel_y + visible_panel_height - arrow_zone_height
            down_arrow_y2 = panel_y + visible_panel_height

            # Entire bounding box for the panel
            overall_top = top_bar_y1
            overall_bottom = panel_y + visible_panel_height
            if (panel_x <= x <= panel_x + panel_width and
                overall_top <= y <= overall_bottom):

                # 2a) If clicked drag bar
                if top_bar_y1 <= y <= top_bar_y2:
                    panel_drag = True
                    return

                # 2b) If clicked UP arrow
                if up_arrow_y1 <= y <= up_arrow_y2:
                    # Move content up by one button => content goes offset down
                    panel_scroll_offset += SCROLL_STEP
                    return

                # 2c) If clicked DOWN arrow
                if down_arrow_y1 <= y <= down_arrow_y2:
                    # Move content down => offset goes negative
                    panel_scroll_offset -= SCROLL_STEP
                    return

                # 2d) Possibly a filter in the scroll region
                scroll_area_top = up_arrow_y2
                scroll_area_bottom = down_arrow_y1

                # We'll loop filters to see if the user clicked any
                current_y = scroll_area_top + panel_scroll_offset
                for i, key in enumerate(filter_names):
                    y_pos = current_y + i * (button_height + padding)

                    # Only check if in visible range
                    if y_pos + button_height < scroll_area_top:
                        continue
                    if y_pos > scroll_area_bottom - button_height:
                        break

                    # The button rect
                    if (panel_x <= x <= panel_x + panel_width and
                        y_pos <= y <= y_pos + button_height):
                        active_filter = key
                        current_filter = key
                        return

    elif event == cv2.EVENT_LBUTTONUP:
        panel_drag = False

    elif event == cv2.EVENT_MOUSEMOVE and panel_drag:
        # Move the panel in XY
        # We'll let them move it horizontally & vertically
        # Or you could clamp so it doesn't go off-screen
        panel_x = x - panel_width // 2
        panel_y = y
        if panel_y < 40:  # keep bar from overlapping toggle arrow
            panel_y = 40

################################################################################
#                          APPLY SELECTED FILTER                               #
################################################################################

def apply_filter_logic(frame, iw, ih, face_landmarks):
    """
    Overlays or modifies the frame in-place based on current_filter.
    Ears, hats, glitch, neon, etc. are all in here.
    """
    global current_filter

    # Grab face landmarks
    left_eye     = face_landmarks.landmark[33]
    right_eye    = face_landmarks.landmark[263]
    nose_tip     = face_landmarks.landmark[1]
    forehead     = face_landmarks.landmark[10]
    top_lip      = face_landmarks.landmark[13]
    bottom_lip   = face_landmarks.landmark[14]

    left_eye_x  = int(left_eye.x * iw)
    left_eye_y  = int(left_eye.y * ih)
    right_eye_x = int(right_eye.x * iw)
    right_eye_y = int(right_eye.y * ih)
    nose_x      = int(nose_tip.x * iw)
    nose_y      = int(nose_tip.y * ih)
    forehead_x  = int(forehead.x * iw)
    forehead_y  = int(forehead.y * ih)
    top_lip_y   = int(top_lip.y * ih)
    bottom_lip_y= int(bottom_lip.y * ih)

    ear_width  = int((right_eye_x - left_eye_x) * 1.8)
    ear_height = int(ear_width * 0.75)

    filter_data = filters[current_filter]

    # ------------------- VINTAGE ------------------- #
    if current_filter == "vintage":
        sepia = cv2.transform(
            frame,
            np.array([
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189]
            ])
        )
        frame[:] = np.clip(sepia, 0, 255).astype(np.uint8)

    # ------------------- GLASSES ------------------- #
    elif current_filter == "glasses":
        if "glass" in filter_data:
            glass = filter_data["glass"]
            glass_width  = int((right_eye_x - left_eye_x) * 2.0)
            glass_height = int(glass_width * 0.4)
            glass_resized = cv2.resize(glass, (glass_width, glass_height), interpolation=cv2.INTER_CUBIC)
            glass_rgb  = glass_resized[:, :, :3]
            glass_mask = glass_resized[:, :, 3]
            gx = left_eye_x - int(glass_width * 0.25)
            gy = left_eye_y - int(glass_height / 2)
            if 0 <= gx < iw - glass_width and 0 <= gy < ih - glass_height:
                roi = frame[gy:gy + glass_height, gx:gx + glass_width]
                if roi.size > 0:
                    frame[gy:gy + glass_height, gx:gx + glass_width] = overlay_image(roi, glass_rgb, glass_mask)

    # ------------------- NEON GLOW ------------------- #
    elif current_filter == "neon_glow":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        _, edges = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        frame[:] = cv2.addWeighted(frame, 0.7, edges, 0.3, 0)

    # ------------------- BLACK & WHITE ------------------- #
    elif current_filter == "Black & white":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame[:] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ------------------- CARTOON ------------------- #
    elif current_filter == "cartoon":
        color = frame.copy()
        for _ in range(3):
            color = cv2.bilateralFilter(color, d=9, sigmaColor=10, sigmaSpace=40)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, blockSize=7, C=2
        )
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color, edges)
        kernel = np.array([[-1, -1, -1],
                           [-1, 30, -1],
                           [-1, -1, -1]])
        frame[:] = cv2.filter2D(cartoon, -1, kernel)

    # ------------------- BEAUTY ------------------- #
    elif current_filter == "beauty":
        blur = cv2.bilateralFilter(frame, d=10, sigmaColor=5, sigmaSpace=20)
        hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 20)
        s = cv2.add(s, 20)
        final_hsv = cv2.merge((h, s, v))
        improved  = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        nicer     = cv2.addWeighted(improved, 1.1, improved, 0, 5)
        frame[:]  = nicer

    # ------------------- NEGATIVE ------------------- #
    elif current_filter == "negative":
        frame[:] = cv2.bitwise_not(frame)

    # ------------------- SEPIA ------------------- #
    elif current_filter == "sepia":
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        tinted = cv2.transform(frame, sepia_filter)
        frame[:] = np.clip(tinted, 0, 255).astype(np.uint8)

    # ------------------- BRIGHTEN ------------------- #
    elif current_filter == "brighten":
        temp = cv2.bilateralFilter(frame, d=2, sigmaColor=20, sigmaSpace=20)
        hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 30)
        final_hsv = cv2.merge((h, s, v))
        frame[:] = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # ------------------- GLITCH ------------------- #
    elif current_filter == "glitch":
        hh, ww, _ = frame.shape
        glitch_frame = frame.copy()
        for i_col in range(0, ww, random.randint(20, 100)):
            y_offset = random.randint(-10, 10)
            glitch_frame[:, i_col:i_col+20] = np.roll(glitch_frame[:, i_col:i_col+20], y_offset, axis=0)
        frame[:] = glitch_frame

    # ------------------- SKETCH ------------------- #
    elif current_filter == "sketch":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted_gray = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted_gray, (111, 111), 0)
        inverted_blurred = cv2.bitwise_not(blurred)
        sk = cv2.divide(gray, inverted_blurred, scale=256.0)
        frame[:] = cv2.cvtColor(sk, cv2.COLOR_GRAY2BGR)

    # ------------------- COWBOY ------------------- #
    elif current_filter == "cowboy":
        hat_img = filter_data.get("hat", None)
        if hat_img is not None:
            hat_width  = int((right_eye_x - left_eye_x) * 2.5)+40
            hat_height = int(hat_width * 0.6) + 20
            hat_resized = cv2.resize(hat_img, (hat_width, hat_height), interpolation=cv2.INTER_CUBIC)
            hat_rgb  = hat_resized[:, :, :3]
            hat_mask = hat_resized[:, :, 3]
            hx = forehead_x - hat_width // 2
            hy = forehead_y - hat_height + 10
            if 0 <= hx < iw - hat_width and 0 <= hy < ih - hat_height:
                roi = frame[hy:hy + hat_height, hx:hx + hat_width]
                if roi.size > 0:
                    frame[hy:hy + hat_height, hx:hx + hat_width] = overlay_image(roi, hat_rgb, hat_mask)

        glass = filter_data.get("glasses", None)
        if glass is not None:
            glass_width  = int((right_eye_x - left_eye_x) * 2.0)
            glass_height = int(glass_width * 0.4) + 40
            glass_resized = cv2.resize(glass, (glass_width, glass_height), interpolation=cv2.INTER_CUBIC)
            glass_rgb  = glass_resized[:, :, :3]
            glass_mask = glass_resized[:, :, 3]
            gx = left_eye_x - int(glass_width * 0.25)
            gy = left_eye_y - int(glass_height / 2) + 10
            if 0 <= gx < iw - glass_width and 0 <= gy < ih - glass_height:
                roi = frame[gy:gy + glass_height, gx:gx + glass_width]
                if roi.size > 0:
                    frame[gy:gy + glass_height, gx:gx + glass_width] = overlay_image(roi, glass_rgb, glass_mask)

        mustache_img = filter_data.get("mustache", None)
        if mustache_img is not None:
            mustache_width = int((right_eye_x - left_eye_x) * 1.2)+40
            mustache_height= int(mustache_width * 0.3) + 20
            mustache_resized = cv2.resize(mustache_img, (mustache_width, mustache_height), interpolation=cv2.INTER_CUBIC)
            mustache_rgb  = mustache_resized[:, :, :3]
            mustache_mask = mustache_resized[:, :, 3]
            mx = nose_x - mustache_width // 2
            my = nose_y -3
            if 0 <= mx < iw - mustache_width and 0 <= my < ih - mustache_height:
                roi = frame[my:my + mustache_height, mx:mx + mustache_width]
                if roi.size > 0:
                    frame[my:my + mustache_height, mx:mx + mustache_width] = overlay_image(roi, mustache_rgb, mustache_mask)

    # ------------------- CAT / DOG ------------------- #
    elif current_filter in ["cat", "dog"]:
        # If dog => check mouth open => add tongue
        if current_filter == "dog":
            mouth_open = (bottom_lip_y - top_lip_y) > 15
            if mouth_open and "tongue" in filter_data:
                tongue_img = filter_data["tongue"]
                tongue_width  = int(ear_width * 0.6)
                tongue_height = int(tongue_width * 0.6)
                tx = int((left_eye_x + right_eye_x) / 2) - tongue_width // 2
                ty = nose_y + int(tongue_height * 0.6)
                tongue_resized = cv2.resize(tongue_img, (tongue_width, tongue_height), interpolation=cv2.INTER_CUBIC)
                tongue_rgb  = tongue_resized[:, :, :3]
                tongue_mask = tongue_resized[:, :, 3]
                if 0 <= tx < iw - tongue_width and 0 <= ty < ih - tongue_height:
                    roi = frame[ty:ty + tongue_height, tx:tx + tongue_width]
                    if roi.size > 0:
                        frame[ty:ty + tongue_height, tx:tx + tongue_width] = overlay_image(roi, tongue_rgb, tongue_mask)

        # Ears
        for part, x_offset in [("left_ear", -ear_width), ("right_ear", 0)]:
            if part in filter_data:
                part_img = filter_data[part]
                resized = cv2.resize(part_img, (ear_width, ear_height), interpolation=cv2.INTER_CUBIC)
                rgb  = resized[:, :, :3]
                mask = resized[:, :, 3]
                ex = forehead_x + x_offset
                ey = forehead_y - int(ear_height * 0.9)
                if 0 <= ex < iw - ear_width and 0 <= ey < ih - ear_height:
                    roi = frame[ey:ey + ear_height, ex:ex + ear_width]
                    if roi.size > 0:
                        frame[ey:ey + ear_height, ex:ex + ear_width] = overlay_image(roi, rgb, mask)

        # Nose
        if "nose" in filter_data:
            nose_img = filter_data["nose"]
            nose_width  = int(ear_width * 0.7)
            nose_height = int(nose_width * 0.5)
            nose_resized = cv2.resize(nose_img, (nose_width, nose_height), interpolation=cv2.INTER_CUBIC)
            nose_rgb  = nose_resized[:, :, :3]
            nose_mask = nose_resized[:, :, 3]
            nx = int((left_eye_x + right_eye_x) / 2) - nose_width // 2
            ny = nose_y - nose_height // 2
            if 0 <= nx < iw - nose_width and 0 <= ny < ih - nose_height:
                roi = frame[ny:ny + nose_height, nx:nx + nose_width]
                if roi.size > 0:
                    frame[ny:ny + nose_height, nx:nx + nose_width] = overlay_image(roi, nose_rgb, nose_mask)

    # ------------------- ORIGINAL ------------------- #
    elif current_filter == "Original":
        pass  # no effect

################################################################################
#                                 MAIN LOOP                                    #
################################################################################

cv2.namedWindow("Face Filter App 😺🐶🤓")
cv2.setMouseCallback("Face Filter App 😺🐶🤓", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # fallback in case of error

    # Flip horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(rgb_frame)
    results_hand = hands.process(rgb_frame)

    ih, iw, _ = frame.shape

    # If filter changed to/from cowboy, handle music
    # last_filterglobal 
    if current_filter != last_filter:
        pygame.mixer.stop()
        if current_filter == "cowboy":
            cowboy_music.play(-1)
        last_filter = current_filter

    # HAND RAISE DETECTION
    hand_is_raised = False
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            wrist_y  = hand_landmarks.landmark[0].y  * ih
            index_y  = hand_landmarks.landmark[8].y  * ih
            if index_y < wrist_y:
                hand_is_raised = True

    # FACE => apply filter
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            apply_filter_logic(frame, iw, ih, face_landmarks)

    # COUNTDOWN / COOLDOWN
    current_time = time.time()
    # global timer_started, countdown_start, cooldown, cooldown_start
    # global last_beep_time

    if hand_is_raised and not timer_started and not cooldown:
        timer_started = True
        countdown_start = current_time
        last_beep_time = -1

    if timer_started:
        elapsed = int(current_time - countdown_start)
        time_left = COUNTDOWN_TIME - elapsed
        if time_left > 0:
            cv2.putText(frame, str(time_left), (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (0, 0, 255), 5)
            if elapsed != last_beep_time:
                if not pygame.mixer.get_busy():
                    beep.play()
                last_beep_time = elapsed
        else:
            # time up => snap a photo
            save_picture(frame)
            timer_started = False
            cooldown = True
            cooldown_start = current_time

    if cooldown and (current_time - cooldown_start > COOLDOWN_TIME):
        cooldown = False

    # UI: Draw panel & toggle
    draw_filter_panel(frame)
    draw_toggle_button(frame)

    # Show
    cv2.imshow("Face Filter App 😺🐶🤓", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Press 's' to save a picture manually
    if key == ord('s'):
        save_picture(frame)

cap.release()
cv2.destroyAllWindows()
