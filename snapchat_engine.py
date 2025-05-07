import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time
import random
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Face Mesh and Hands
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(max_num_hands=1)

# Global variables
BASE_DIR = r"D:\LA-FINAL-PROJECT"
PIC_DIR = os.path.join(BASE_DIR, "static", "Pictures")
os.makedirs(PIC_DIR, exist_ok=True)

pygame.init()
beep = pygame.mixer.Sound(os.path.join(BASE_DIR, "Beep.wav"))
click = pygame.mixer.Sound(os.path.join(BASE_DIR, "iphone-camera-capture-6448.wav"))
cowboy_music = pygame.mixer.Sound(os.path.join(BASE_DIR, "cowboy_music.wav"))
chath_music = pygame.mixer.Sound(os.path.join(BASE_DIR, "BADO BADI.wav"))

# Try multiple camera indices to ensure webcam access
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        logger.debug(f"Webcam opened successfully on index {i}")
        break
else:
    logger.error("Unable to open any webcam")
    cap = None

# Load filter assets
def png(path):
    full_path = os.path.join(BASE_DIR, path)
    if not os.path.exists(full_path):
        logger.error(f"Image file not found: {full_path}")
        return None
    img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.error(f"Failed to load image: {full_path}")
        return None
    if img.shape[2] != 4:
        logger.error(f"Image does not have an alpha channel: {full_path}")
        return None
    return img

leaf_images = [
    png("Leave 1.png"),
    png("leave 2.png"),
    png("leave 3.png"),
]

filters = {
    "cat": {
        "left_ear": png("left.png"),
        "right_ear": png("right.png"),
        "nose": png("nose.png"),
    },
    "dog": {
        "left_ear": png("Dogleft.png"),
        "right_ear": png("Dogright.png"),
        "nose": png("Dognose.png"),
        "tongue": png("tongue.png"),
    },
    "glasses": {
        "glass": png("Glasses.png"),
    },
    "cowboy": {
        "hat": png("cowboy_hat.png"),
        "mustache": png("moustache.png"),
    },
}

# Load CHATH sticker
CH = png("CHATH.png")
if CH is not None:
    alpha = CH[:, :, 3]
    mask_ch = (alpha > 200).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask_ch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(cnts[0])
        filters["chath"] = {
            "rgb": CH[y:y+h, x:x+w, :3],
            "mask": mask_ch[y:y+h, x:x+w]
        }
        logger.debug(f"CHATH loaded: w={w}, h={h}")
    else:
        logger.error("No contours found in CHATH.png mask")
        filters["chath"] = {}
else:
    filters["chath"] = {}

# Load FASIAL sticker
FASIAL_PATH = os.path.join(BASE_DIR, "testingf2.png")
fasial_img = cv2.imread(FASIAL_PATH, cv2.IMREAD_UNCHANGED)
if fasial_img is None:
    logger.error(f"Error: testingf2.png not found or could not be loaded.")
else:
    alpha_fasial = fasial_img[:, :, 3]
    mask_fasial = (alpha_fasial > 200).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask_fasial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(cnts[0])
        filters["fasial"] = {
            "rgb": fasial_img[y:y+h, x:x+w, :3],
            "mask": mask_fasial[y:y+h, x:x+w],
            "gradient": None,
            "silver_gradient": None,
            "stars": None
        }
        logger.debug(f"Testingf2 loaded: w={w}, h={h}")
    else:
        logger.error("No contours found in testingf2.png mask")
        filters["fasial"] = {}

# Effect filters
effect_names = [
    "vintage", "mirror", "autumn", "snow", "pixelate", "old_film", "thermal",
    "galaxy", "brighten", "Black & white", "cartoon", "beauty", "sketch",
    "negative", "sepia", "glitch", "neon_glow", "Original", "fasial"
]
for name in effect_names:
    if name not in filters:
        filters[name] = {}

galaxy_img = png("galaxy.png")
if galaxy_img is None:
    galaxy_img = np.zeros((480, 640, 3), dtype=np.uint8)

filter_names = list(filters.keys())
current_filter = filter_names[0]
last_filter = None
intensity = 100
snowflakes = []
leaves = []

def init_snowflakes(num_flakes, iw, ih):
    global snowflakes
    snowflakes = []
    for _ in range(num_flakes):
        snowflakes.append({
            'x': np.random.randint(0, iw),
            'y': np.random.randint(-ih, ih),
            'vy': np.random.uniform(2, 5),
            'size': np.random.randint(2, 5)
        })

def init_leaves(num_leaves, width, height):
    global leaves
    leaves = []
    for _ in range(num_leaves):
        side = random.choice(['top', 'left', 'right'])
        if side == 'top':
            x = random.uniform(0, width)
            y = random.uniform(-height * 0.2, 0)
            vx = random.uniform(-0.5, 0.5)
            vy = random.uniform(5, 10)
        elif side == 'left':
            x = random.uniform(-width * 0.2, 0)
            y = random.uniform(0, height)
            vx = random.uniform(0, 1)
            vy = random.uniform(5, 10)
        else:
            x = random.uniform(width, width * 1.2)
            y = random.uniform(0, height)
            vx = random.uniform(-1, 0)
            vy = random.uniform(5, 10)
        leaves.append({
            'img': random.choice(leaf_images),
            'x': x, 'y': y,
            'vx': vx, 'vy': vy,
            'angle': random.uniform(0, 360),
            'angular_velocity': random.uniform(-5, 5),
            'scale': random.uniform(0.1, 0.2)
        })

# Precompute dark theme background for fasial filter
def create_dark_theme_background(iw, ih):
    gradient = np.zeros((ih, iw, 3), dtype=np.uint8)
    for i in range(ih):
        t = i / ih
        color = (int(30 * (1 - t) + 10 * t), int(30 * (1 - t) + 10 * t), int(40 * (1 - t) + 20 * t))
        gradient[i, :] = color
    silver_gradient = np.zeros((ih, iw, 3), dtype=np.uint8)
    for i in range(ih):
        t = i / ih
        silver = int(50 * (1 - t))
        silver_gradient[i, :] = (silver, silver, silver)
    stars = np.zeros((ih, iw, 3), dtype=np.uint8)
    num_stars = 50
    for _ in range(num_stars):
        x = np.random.randint(0, iw)
        y = np.random.randint(0, ih)
        brightness = np.random.randint(100, 255)
        cv2.circle(stars, (x, y), 1, (brightness, brightness, brightness), -1)
    return gradient, silver_gradient, stars

def apply_dark_theme_background(frame, gradient, silver_gradient, stars):
    frame[:] = cv2.addWeighted(frame, 0.6, gradient, 0.4, 0)
    frame[:] = cv2.addWeighted(frame, 0.8, silver_gradient, 0.2, 0)
    frame[:] = cv2.addWeighted(frame, 0.9, stars, 0.1, 0)
    return frame

def overlay_image(roi, img, m):
    try:
        img = img.astype(np.uint8)
        m = m.astype(np.uint8)
        if roi.shape[:2] != img.shape[:2]:
            img = cv2.resize(img, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)
            m = cv2.resize(m, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        inv = cv2.bitwise_not(m)
        bg = cv2.bitwise_and(roi, roi, mask=inv)
        fg = cv2.bitwise_and(img, img, mask=m)
        return cv2.add(bg, fg)
    except Exception as e:
        logger.error(f"Overlay image error: {e}")
        return roi

def get_face_box(landmarks, w, h, vertical_offset=60):  # Increased vertical_offset to 60
    if not landmarks:
        logger.warning("No face landmarks detected")
        return 0, 0, w, h
    pts = np.array([[p.x*w, p.y*h] for p in landmarks.landmark])
    hull_idx = [10, 152, 234, 454]
    hull = pts[hull_idx]
    x, y, bw, bh = cv2.boundingRect(hull.astype(np.float32))
    pad_w, pad_h = int(bw*0.3), int(bh*0.6)  # Increased pad_h to 0.6 to extend downward
    x0 = max(0, x-pad_w)
    y0 = max(0, y-pad_h-vertical_offset)
    return x0, y0, min(w, x+bw+pad_w)-x0, min(h, y+bh+pad_h)-y0  # Adjusted min(h, y+bh+pad_h) to extend further

def apply_filter_logic(frame, iw, ih, landmarks):
    global current_filter, intensity, leaves
    intensity_factor = intensity / 100.0
    fdata = filters[current_filter]

    if landmarks:
        le, re = landmarks.landmark[33], landmarks.landmark[263]
        nt, fh = landmarks.landmark[1], landmarks.landmark[10]
        tl, bl = landmarks.landmark[13], landmarks.landmark[14]
        lex, ley = int(le.x*iw), int(le.y*ih)
        rex, rey = int(re.x*iw), int(re.y*ih)
        nx, ny = int(nt.x*iw), int(nt.y*ih)
        fx, fy = int(fh.x*iw), int(fh.y*ih)
        tly, bly = int(tl.y*ih), int(bl.y*ih)
        ew = int((rex-lex)*1.8)
        eh = int(ew*0.75)
    else:
        lex, ley, rex, rey, nx, ny, fx, fy, tly, bly = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ew, eh = iw // 4, ih // 4

    try:
        if current_filter == "vintage":
            M = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
            sep = cv2.transform(frame, M)
            frame[:] = np.clip(sep, 0, 255).astype(np.uint8)

        elif current_filter == "glasses" and "glass" in fdata and fdata["glass"] is not None:
            g = fdata["glass"]
            gw, gh = int((rex-lex)*2), int((rex-lex)*0.4)+80
            r = cv2.resize(g, (gw,gh), interpolation=cv2.INTER_CUBIC)
            rgb, mask = r[:,:,:3], r[:,:,3]
            gx, gy = lex-int(gw*0.25), ley-gh//2+10
            if 0<=gx<iw-gw and 0<=gy<ih-gh:
                roi = frame[gy:gy+gh, gx:gx+gw]
                frame[gy:gy+gh, gx:gx+gw] = overlay_image(roi, rgb, mask)

        elif current_filter == "neon_glow":
            gray = cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(21,21),0)
            _, edges = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            frame[:] = cv2.addWeighted(frame,0.7,edges,0.3,0)

        elif current_filter == "Black & white":
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame[:] = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

        elif current_filter == "cartoon":
            color = frame.copy()
            for _ in range(3):
                color = cv2.bilateralFilter(color,9,10,40)
            gray = cv2.medianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),7)
            edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,2)
            edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
            cartoon = cv2.bitwise_and(color, edges)
            kernel = np.array([[-1,-1,-1],[-1,30,-1],[-1,-1,-1]])
            kernel = kernel / np.sum(kernel)
            cartoon = cv2.filter2D(cartoon, -1, kernel)
            frame[:] = np.clip(cartoon, 0, 255).astype(np.uint8)

        elif current_filter == "beauty":
            sigma = 75 * intensity_factor
            bil = cv2.bilateralFilter(frame,d=15,sigmaColor=int(sigma),sigmaSpace=75)
            frame[:] = cv2.addWeighted(bil,0.6,frame,0.4,0)

        elif current_filter == "brighten":
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(hsv)
            v = np.clip(v+30,0,255).astype(np.uint8)
            frame[:] = cv2.cvtColor(cv2.merge((h,s,v)),cv2.COLOR_HSV2BGR)

        elif current_filter == "pixelate":
            h, w = frame.shape[:2]
            block = int(7 * (0.5 + intensity_factor))
            small = cv2.resize(frame, (w//block,h//block), interpolation=cv2.INTER_LINEAR)
            frame[:] = cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST)

        elif current_filter == "mirror":
            frame[:] = cv2.flip(frame,1)

        elif current_filter == "old_film":
            M = np.array([[0.393,0.769,0.189],[0.349,0.686,0.168],[0.272,0.534,0.131]])
            sepia = cv2.transform(frame, M)
            sepia = np.clip(sepia,0,255).astype(np.uint8)
            noise = np.random.normal(0,2,sepia.shape).astype(np.uint8)
            frame[:] = np.clip(cv2.add(sepia, noise), 0, 255).astype(np.uint8)

        elif current_filter == "thermal":
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame[:] = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        elif current_filter == "galaxy":
            gal = cv2.resize(galaxy_img,(iw,ih), interpolation=cv2.INTER_AREA)
            frame[:] = cv2.addWeighted(frame,0.3,gal,0.7,0)

        elif current_filter == "negative":
            frame[:] = cv2.bitwise_not(frame)

        elif current_filter == "sepia":
            M = np.array([[0.393,0.769,0.189],[0.349,0.686,0.168],[0.272,0.534,0.131]]) * intensity_factor
            tinted = cv2.transform(frame, M)
            frame[:] = np.clip(tinted,0,255).astype(np.uint8)

        elif current_filter == "glitch":
            ww = frame.shape[1]
            for i in range(0, ww, random.randint(20,100)):
                off = random.randint(-10,10)
                frame[:, i:i+20] = np.roll(frame[:, i:i+20], off, axis=0)

        elif current_filter == "sketch":
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            inv = cv2.bitwise_not(gray)
            blur = cv2.GaussianBlur(inv,(111,111),0)
            sk = cv2.divide(gray, cv2.bitwise_not(blur), scale=256.0)
            frame[:] = cv2.cvtColor(sk, cv2.COLOR_GRAY2BGR)

        elif current_filter in ["chath", "fasial"] and "rgb" in fdata and "mask" in fdata:
            logger.debug(f"Applying filter: {current_filter}")
            if current_filter == "chath":
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hue = int(time.time() * 60) % 180
                rain = np.full_like(hsv, (hue, 255, 255))
                rr = cv2.cvtColor(rain, cv2.COLOR_HSV2BGR)
                frame[:] = cv2.addWeighted(frame, 0.5, rr, 0.5, 0)

            x0, y0, bw, bh = get_face_box(landmarks, iw, ih)
            logger.debug(f"Face box: x0={x0}, y0={y0}, bw={bw}, bh={bh}")
            if x0 >= 0 and y0 >= 0 and bw > 0 and bh > 0:
                try:
                    current_size = (bw, bh)
                    if "last_size" not in fdata or fdata["last_size"] != current_size or fdata.get("cached_donor") is None or fdata.get("cached_mask") is None:
                        fdata["cached_donor"] = cv2.resize(fdata["rgb"], (bw, bh), interpolation=cv2.INTER_AREA)
                        fdata["cached_mask"] = cv2.resize(fdata["mask"], (bw, bh), interpolation=cv2.INTER_NEAREST)
                        fdata["last_size"] = current_size
                    
                    h_clip = min(bh, ih - y0)
                    w_clip = min(bw, iw - x0)
                    logger.debug(f"Clip dimensions: h_clip={h_clip}, w_clip={w_clip}")
                    if h_clip > 0 and w_clip > 0 and fdata["cached_donor"].shape[0] >= h_clip and fdata["cached_donor"].shape[1] >= w_clip and fdata["cached_mask"].shape[0] >= h_clip and fdata["cached_mask"].shape[1] >= w_clip:
                        donor = fdata["cached_donor"][:h_clip, :w_clip]
                        msk = fdata["cached_mask"][:h_clip, :w_clip]
                        roi = frame[y0:y0+h_clip, x0:x0+w_clip]
                        frame[y0:y0+h_clip, x0:x0+w_clip] = overlay_image(roi, donor, msk)
                        logger.debug(f"{current_filter.capitalize()} applied at x0={x0}, y0={y0}, bw={bw}, bh={bh}")
                    else:
                        logger.error(f"Invalid clip dimensions for {current_filter}: h_clip={h_clip}, w_clip={w_clip}")
                except Exception as e:
                    logger.error(f"Error applying {current_filter} overlay: {e}")

            if current_filter == "fasial" and fdata["gradient"] is None:
                fdata["gradient"], fdata["silver_gradient"], fdata["stars"] = create_dark_theme_background(iw, ih)
                frame = apply_dark_theme_background(frame, fdata["gradient"], fdata["silver_gradient"], fdata["stars"])

        elif current_filter == "snow":
            if not snowflakes:
                init_snowflakes(100, iw, ih)
            snow_layer = np.zeros_like(frame)
            for flake in snowflakes:
                flake['y'] += flake['vy'] * intensity_factor
                if flake['y'] > ih:
                    flake['x'] = np.random.randint(0, iw)
                    flake['y'] = np.random.randint(-ih, 0)
                center = (int(flake['x']), int(flake['y']))
                radius = int(flake['size'])
                if 0 <= center[0] < iw and 0 <= center[1] < ih:
                    cv2.circle(snow_layer, center, radius, (255, 255, 255), -1)
            alpha = 0.5
            frame[:] = cv2.addWeighted(snow_layer, alpha, frame, 1 - alpha, 0)

        elif current_filter == "autumn":
            if not leaves:
                init_leaves(30, iw, ih)
            overlay = frame.copy()
            for leaf in leaves:
                leaf['x'] += leaf['vx'] * intensity_factor
                leaf['y'] += leaf['vy'] * intensity_factor
                leaf['angle'] += leaf['angular_velocity']
                if (leaf['y'] > ih + 50 or leaf['x'] < -50 or leaf['x'] > iw + 50):
                    side = random.choice(['top', 'left', 'right'])
                    if side == 'top':
                        leaf['x'], leaf['y'] = random.uniform(0, iw), random.uniform(-ih*0.2, 0)
                        leaf['vx'], leaf['vy'] = random.uniform(-0.5, 0.5), random.uniform(5, 10)
                    elif side == 'left':
                        leaf['x'], leaf['y'] = random.uniform(-iw*0.2, 0), random.uniform(0, ih)
                        leaf['vx'], leaf['vy'] = random.uniform(0, 1), random.uniform(5, 10)
                    else:
                        leaf['x'], leaf['y'] = random.uniform(iw, iw*1.2), random.uniform(0, ih)
                        leaf['vx'], leaf['vy'] = random.uniform(-1, 0), random.uniform(5, 10)
                    leaf['angle'] = random.uniform(0, 360)
                    leaf['angular_velocity'] = random.uniform(-5, 5)
                    leaf['scale'] = random.uniform(0.1, 0.2)
                img = leaf['img']
                if img is None:
                    continue
                h0, w0 = img.shape[:2]
                sw, sh = int(w0 * leaf['scale']), int(h0 * leaf['scale'])
                resized = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
                M = cv2.getRotationMatrix2D((sw/2, sh/2), leaf['angle'], 1)
                rotated = cv2.warpAffine(
                    resized, M, (sw, sh),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0,0,0,0)
                )
                x, y = int(leaf['x']), int(leaf['y'])
                y1, y2 = max(0, y), min(ih, y + sh)
                x1, x2 = max(0, x), min(iw, x + sw)
                ly1, ly2 = max(0, -y), min(sh, ih - y)
                lx1, lx2 = max(0, -x), min(sw, iw - x)
                if y1 < y2 and x1 < x2:
                    alpha = rotated[ly1:ly2, lx1:lx2, 3:] / 255.0
                    fg = rotated[ly1:ly2, lx1:lx2, :3]
                    bg = overlay[y1:y2, x1:x2]
                    overlay[y1:y2, x1:x2] = (alpha * fg + (1 - alpha) * bg).astype(np.uint8)
            frame[:] = overlay

        elif current_filter == "cowboy":
            hat = fdata.get("hat")
            ms = fdata.get("mustache")
            if hat is not None:
                hw = int((rex-lex)*2.5)+40
                hh_ = int(hw*0.6)+20
                hx = fx-hw//2
                hy = fy-hh_+10
                if 0<=hx<iw-hw and 0<=hy<ih-hh_:
                    r = cv2.resize(hat,(hw,hh_),cv2.INTER_CUBIC)
                    frame[hy:hy+hh_, hx:hx+hw] = overlay_image(frame[hy:hy+hh_, hx:hx+hw], r[:,:,:3], r[:,:,3])
            if ms is not None:
                mw = int((rex-lex)*1.2)+40
                mh = int(mw*0.3)+20
                mx = nx-mw//2
                my = ny-3
                if 0<=mx<iw-mw and 0<=my<ih-mh:
                    m_img = cv2.resize(ms,(mw,mh),cv2.INTER_CUBIC)
                    frame[my:my+mh, mx:mx+mw] = overlay_image(frame[my:my+mh, mx:mx+mw], m_img[:,:,:3], m_img[:,:,3])

        elif current_filter in ("cat","dog"):
            fdata = filters[current_filter]
            if current_filter == "dog" and (bly - tly) > 15 and "tongue" in fdata and fdata["tongue"] is not None:
                tw, th = int(ew*0.6), int(ew*0.6*0.6)
                tx = (lex+rex)//2 - tw//2
                ty = ny + int(th*0.6)
                if 0<=tx<iw-tw and 0<=ty<ih-th:
                    t_img = cv2.resize(fdata["tongue"], (tw,th), cv2.INTER_CUBIC)
                    frame[ty:ty+th, tx:tx+tw] = overlay_image(frame[ty:ty+th, tx:tx+tw], t_img[:,:,:3], t_img[:,:,3])
            for part,off in [("left_ear",-ew),("right_ear",0)]:
                if part in fdata and fdata[part] is not None:
                    ex = fx + off
                    ey = fy - int(eh*0.9)
                    if 0<=ex<iw-ew and 0<=ey<ih-eh:
                        e_img = cv2.resize(fdata[part], (ew,eh), cv2.INTER_CUBIC)
                        frame[ey:ey+eh, ex:ex+ew] = overlay_image(frame[ey:ey+eh, ex:ex+ew], e_img[:,:,:3], e_img[:,:,3])
            if "nose" in fdata and fdata["nose"] is not None:
                nw, nh = int(ew*0.7), int(ew*0.5*0.5)
                nxp = (lex+rex)//2 - nw//2
                nyp = ny - nh//2
                if 0<=nxp<iw-nw and 0<=nyp<ih-nh:
                    n_img = cv2.resize(fdata["nose"], (nw,nh), cv2.INTER_CUBIC)
                    frame[nyp:nyp+nh, nxp:nxp+nw] = overlay_image(frame[nyp:nyp+nh, nxp:nxp+nw], n_img[:,:,:3], n_img[:,:,3])

    except Exception as e:
        logger.error(f"Error applying filter {current_filter}: {e}")

# Server API
def gen_frames():
    global last_filter, frame_counter, last_landmarks
    frame_counter = 0
    last_landmarks = None
    while True:
        if cap is None or not cap.isOpened():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Error: Cannot access webcam", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            logger.error("Webcam not accessible, displaying error message")
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error("Failed to capture frame")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Error: Failed to capture frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                time.sleep(0.1)
            else:
                frame = cv2.flip(frame, 1)
                logger.debug(f"Frame captured: shape={frame.shape}")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ih, iw, _ = frame.shape

                if current_filter != last_filter:
                    pygame.mixer.stop()
                    if current_filter == "cowboy":
                        cowboy_music.play(-1)
                    if current_filter == "chath":
                        chath_music.play(-1)
                    last_filter = current_filter

                frame_counter += 1
                if frame_counter % 5 == 0 or last_landmarks is None:
                    results = face_mesh.process(rgb)
                    if results.multi_face_landmarks:
                        last_landmarks = results.multi_face_landmarks[0]
                    else:
                        last_landmarks = None
                
                apply_filter_logic(frame, iw, ih, last_landmarks)

                frame = np.clip(frame, 0, 255).astype(np.uint8)
                ret2, buf = cv2.imencode(".jpg", frame)
                if not ret2:
                    logger.error("Failed to encode frame")
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )

def set_filter(name):
    global current_filter
    if name in filter_names:
        current_filter = name
        logger.debug(f"Set filter to {name}")

def set_intensity(value):
    global intensity
    try:
        intensity = max(0, min(100, int(value)))
        logger.debug(f"Set intensity to {intensity}")
    except ValueError:
        logger.error(f"Invalid intensity value: {value}")

def save_snapshot(overlay_data=None, save_dir=PIC_DIR):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            if cap is None or not cap.isOpened():
                logger.error("Cannot save snapshot: Webcam not accessible")
                return None
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                logger.error(f"Attempt {attempt+1}: Failed to capture frame for snapshot")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            ih, iw = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_mesh.process(rgb)

            if faces.multi_face_landmarks:
                apply_filter_logic(frame, iw, ih, faces.multi_face_landmarks[0])
            else:
                apply_filter_logic(frame, iw, ih, None)

            if overlay_data is not None:
                try:
                    overlay = cv2.imdecode(np.frombuffer(overlay_data, np.uint8), cv2.IMREAD_UNCHANGED)
                    if overlay is None:
                        logger.warning("Overlay data is invalid, skipping overlay")
                    else:
                        if overlay.shape[:2] != frame.shape[:2]:
                            overlay = cv2.resize(overlay, (iw, ih), interpolation=cv2.INTER_CUBIC)
                        if overlay.shape[2] == 4:
                            alpha = overlay[:, :, 3] / 255.0
                            for c in range(3):
                                frame[:, :, c] = frame[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha
                        else:
                            frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
                except Exception as e:
                    logger.error(f"Overlay compositing error: {e}")

            frame = np.clip(frame, 0, 255).astype(np.uint8)
            ts = int(time.time() * 1000)
            fname = f"snap_{ts}.jpg"
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, fname)
            if not cv2.imwrite(path, frame):
                logger.error(f"Attempt {attempt+1}: Failed to save snapshot to {path}")
                continue
            click.play()
            logger.debug(f"Saved snapshot: {path}")
            return fname
        except Exception as e:
            logger.error(f"Attempt {attempt+1}: Error saving snapshot: {e}")
            time.sleep(0.1)
    logger.error(f"Failed to save snapshot after {max_attempts} attempts")
    return None

def record_video(duration=5, save_dir=PIC_DIR):
    fps = 30
    if cap is None or not cap.isOpened():
        logger.error("Cannot record video: Webcam not accessible")
        return None
    ret, frame = cap.read()
    if not ret or frame is None:
        logger.error("Failed to capture frame for video")
        return None
    frame = cv2.flip(frame, 1)
    ih, iw = frame.shape[:2]
    ts = int(time.time() * 1000)
    fname = f"video_{ts}.mp4"
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, fname)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (iw, ih))
    
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame during video recording")
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_mesh.process(rgb)
        if faces.multi_face_landmarks:
            apply_filter_logic(frame, iw, ih, faces.multi_face_landmarks[0])
        else:
            apply_filter_logic(frame, iw, ih, None)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        out.write(frame)
    
    out.release()
    if not os.path.exists(path):
        logger.error(f"Failed to save video: File not created at {path}")
        return None
    logger.debug(f"Saved video: {path}")
    return fname

def record_boomerang(duration=3, save_dir=PIC_DIR):
    fps = 30
    if cap is None or not cap.isOpened():
        logger.error("Cannot record boomerang: Webcam not accessible")
        return None
    ret, frame = cap.read()
    if not ret or frame is None:
        logger.error("Failed to capture frame for boomerang")
        return None
    frame = cv2.flip(frame, 1)
    ih, iw = frame.shape[:2]
    ts = int(time.time() * 1000)
    fname = f"boomerang_{ts}.mp4"
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, fname)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (iw, ih))
    
    frames = []
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame during boomerang recording")
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_mesh.process(rgb)
        if faces.multi_face_landmarks:
            apply_filter_logic(frame, iw, ih, faces.multi_face_landmarks[0])
        else:
            apply_filter_logic(frame, iw, ih, None)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame)
    
    for frame in frames:
        out.write(frame)
    for frame in reversed(frames):
        out.write(frame)
    
    out.release()
    if not os.path.exists(path):
        logger.error(f"Failed to save boomerang: File not created at {path}")
        return None
    logger.debug(f"Saved boomerang: {path}")
    return fname

def list_snapshots(save_dir=PIC_DIR):
    try:
        files = [f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.mp4'))]
        return sorted(files, key=lambda x: os.path.getctime(os.path.join(save_dir, x)), reverse=True)
    except Exception as e:
        logger.error(f"Error listing snapshots in {save_dir}: {e}")
        return []