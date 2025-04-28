import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time
import random



# Example base directory (ensure this is correct)
BASE_DIR = "C:\second semester\Linear Algebra\LA-FINAL-PROJECT"


HERE    = os.path.dirname(os.path.abspath(_file_))
PIC_DIR = os.path.join(HERE, "static", "Pictures")
os.makedirs(PIC_DIR, exist_ok=True)

pygame.init()
beep          = pygame.mixer.Sound(os.path.join(BASE_DIR, "C:\second semester\Linear Algebra\LA-FINAL-PROJECT\Beep.wav"))
click         = pygame.mixer.Sound(os.path.join(BASE_DIR, "C:\second semester\Linear Algebra\LA-FINAL-PROJECT\iphone-camera-capture-6448.wav"))
cowboy_music  = pygame.mixer.Sound(os.path.join(BASE_DIR, "C:\second semester\Linear Algebra\LA-FINAL-PROJECT\cowboy_music.wav"))
chath_music   = pygame.mixer.Sound(os.path.join(BASE_DIR, "C:\second semester\Linear Algebra\LA-FINAL-PROJECT\BADO BADI.wav"))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open webcam")

mp_face_mesh = mp.solutions.face_mesh
mp_hands     = mp.solutions.hands
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(max_num_hands=1)

# ─── LOAD FILTER ASSETS ────────────────────────────────────────────────────────
def png(path):
    return cv2.imread(os.path.join(BASE_DIR, path), cv2.IMREAD_UNCHANGED)

# sticker filters
filters = {
    "cat": {
        "left_ear" : png("C:\second semester\Linear Algebra\LA-FINAL-PROJECT\left.png"),
        "right_ear": png("C:\second semester\Linear Algebra\LA-FINAL-PROJECT\right.png"),
        "nose"     : png("C:\second semester\Linear Algebra\LA-FINAL-PROJECT\nose.png"),
    },
    "dog": {
        "left_ear" : png("C:\second semester\Linear Algebra\LA-FINAL-PROJECT\Dogleft.png"),
        "right_ear": png("C:\second semester\Linear Algebra\LA-FINAL-PROJECT\Dogright.png"),
        "nose"     : png("C:\second semester\Linear Algebra\LA-FINAL-PROJECT\Dognose.png"),
        "tongue"   : png("C:\second semester\Linear Algebra\LA-FINAL-PROJECT\tongue.png"),
    },
    "glasses": {
        "glass": png("C:\second semester\Linear Algebra\LA-FINAL-PROJECT\Glasses.png"),
    },
    "cowboy": {
        "hat"     : png("C:\second semester\Linear Algebra\LA-FINAL-PROJECT\cowboy_hat.png"),
        "mustache": png("C:\second semester\Linear Algebra\LA-FINAL-PROJECT\moustache.png"),
    },
    # chath below
}

# load CHATH sticker
CH = png("C:\second semester\Linear Algebra\LA-FINAL-PROJECT\CHATH.png")
alpha = CH[:, :, 3]
mask_ch = (alpha > 200).astype(np.uint8) * 255
cnts, _ = cv2.findContours(mask_ch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(cnts[0])
filters["chath"] = {
    "rgb":  CH[y:y+h, x:x+w, :3],
    "mask": mask_ch[y:y+h, x:x+w]
}

# effect filters (no images needed)
effect_names = [
    "vintage", "mirror", "pixelate", "old_film", "thermal",
    "galaxy", "brighten", "Black & white", "cartoon",
    "beauty", "sketch", "negative", "sepia", "glitch", "neon_glow", "Original"
]
for name in effect_names:
    filters[name] = {}

# galaxy background for galaxy filter
galaxy_img = png("C:\second semester\Linear Algebra\LA-FINAL-PROJECT\galaxy.jpg")
if galaxy_img is None:
    galaxy_img = np.zeros((480,640,3), dtype=np.uint8)

# build filter list
filter_names    = list(filters.keys())
current_filter  = filter_names[0]
last_filter     = None

# ─── UTILITY FUNCTIONS ────────────────────────────────────────────────────────
def overlay_image(roi, img, m):
    img = img.astype(np.uint8); m = m.astype(np.uint8)
    if roi.shape[:2] != img.shape[:2]:
        img = cv2.resize(img, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)
        m   = cv2.resize(m,   (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    inv = cv2.bitwise_not(m)
    bg  = cv2.bitwise_and(roi, roi, mask=inv)
    fg  = cv2.bitwise_and(img, img, mask=m)
    return cv2.add(bg, fg)

def get_face_box(landmarks, w, h):
    pts = np.array([[p.x*w, p.y*h] for p in landmarks.landmark])
    hull_idx = [10, 152, 234, 454]
    hull = pts[hull_idx]
    x,y,bw,bh = cv2.boundingRect(hull.astype(np.float32))
    pad_w, pad_h = int(bw*0.3), int(bh*0.4)
    x0 = max(0, x-pad_w); y0 = max(0, y-pad_h)
    return x0, y0, min(w, x+bw+pad_w)-x0, min(h, y+bh)-y0

# ─── CORE FILTER LOGIC ────────────────────────────────────────────────────────
def apply_filter_logic(frame, iw, ih, lm):
    global current_filter
    le, re = lm.landmark[33], lm.landmark[263]
    nt, fh = lm.landmark[1],  lm.landmark[10]
    tl, bl = lm.landmark[13], lm.landmark[14]

    lex, ley = int(le.x*iw), int(le.y*ih)
    rex, rey = int(re.x*iw), int(re.y*ih)
    nx, ny   = int(nt.x*iw), int(nt.y*ih)
    fx, fy   = int(fh.x*iw), int(fh.y*ih)
    tly, bly = int(tl.y*ih), int(bl.y*ih)

    ew = int((rex-lex)*1.8)
    eh = int(ew*0.75)
    fdata = filters[current_filter]

    # ───── vintage ─────
    if current_filter == "vintage":
        M = np.array([[0.272,0.534,0.131],
                      [0.349,0.686,0.168],
                      [0.393,0.769,0.189]])
        sep = cv2.transform(frame, M)
        frame[:] = np.clip(sep, 0, 255).astype(np.uint8)

    # ───── glasses ─────
    elif current_filter == "glasses" and "glass" in fdata:
        g = fdata["glass"]
        gw, gh = int((rex-lex)*2)+20, int((rex-lex)*0.4)
        r = cv2.resize(g, (gw,gh), interpolation=cv2.INTER_CUBIC)
        rgb, mask = r[:,:,:3], r[:,:,3]
        gx, gy = lex-int(gw*0.25), ley-gh//2
        if 0<=gx<iw-gw and 0<=gy<ih-gh:
            roi = frame[gy:gy+gh, gx:gx+gw]
            frame[gy:gy+gh, gx:gx+gw] = overlay_image(roi, rgb, mask)

    # ───── neon_glow ─────
    elif current_filter == "neon_glow":
        gray = cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(21,21),0)
        _, edges = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        frame[:] = cv2.addWeighted(frame,0.7,edges,0.3,0)

    # ─── Black & white ───
    elif current_filter == "Black & white":
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame[:] = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

    # ───── cartoon ─────
    elif current_filter == "cartoon":
        color = frame.copy()
        for _ in range(3):
            color = cv2.bilateralFilter(color,9,10,40)
        gray = cv2.medianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),7)
        edges = cv2.adaptiveThreshold(gray,255,
                     cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,7,2)
        edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color, edges)
        kernel = np.array([[-1,-1,-1],[-1,30,-1],[-1,-1,-1]])
        frame[:] = cv2.filter2D(cartoon, -1, kernel)

    # ───── beauty ─────
    elif current_filter == "beauty":
        bil = cv2.bilateralFilter(frame,d=15,sigmaColor=75,sigmaSpace=75)
        frame[:] = cv2.addWeighted(bil,0.6,frame,0.4,0)

    # ───── brighten ─────
    elif current_filter == "brighten":
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        v = np.clip(v+30,0,255).astype(np.uint8)
        frame[:] = cv2.cvtColor(cv2.merge((h,s,v)),cv2.COLOR_HSV2BGR)

    # ───── pixelate ─────
    elif current_filter == "pixelate":
        h, w = frame.shape[:2]
        block = 7
        small = cv2.resize(frame, (w//block,h//block), interpolation=cv2.INTER_LINEAR)
        frame[:] = cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST)

    # ───── mirror ─────
    elif current_filter == "mirror":
        frame[:] = cv2.flip(frame,1)

    # ─── old_film ───
    elif current_filter == "old_film":
        M = np.array([[0.393,0.769,0.189],[0.349,0.686,0.168],[0.272,0.534,0.131]])
        sepia = cv2.transform(frame, M)
        sepia = np.clip(sepia,0,255).astype(np.uint8)
        noise = np.random.normal(0,2,sepia.shape).astype(np.uint8)
        frame[:] = cv2.add(sepia, noise)

    # ───── thermal ─────
    elif current_filter == "thermal":
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame[:] = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # ───── galaxy ─────
    elif current_filter == "galaxy":
        gal = cv2.resize(galaxy_img,(iw,ih), interpolation=cv2.INTER_AREA)
        frame[:] = cv2.addWeighted(frame,0.3,gal,0.7,0)

    # ───── negative ─────
    elif current_filter == "negative":
        frame[:] = cv2.bitwise_not(frame)

    # ───── sepia ─────
    elif current_filter == "sepia":
        M = np.array([[0.393,0.769,0.189],[0.349,0.686,0.168],[0.272,0.534,0.131]])
        tinted = cv2.transform(frame, M)
        frame[:] = np.clip(tinted,0,255).astype(np.uint8)

    # ───── glitch ─────
    elif current_filter == "glitch":
        ww = frame.shape[1]
        for i in range(0, ww, random.randint(20,100)):
            off = random.randint(-10,10)
            frame[:, i:i+20] = np.roll(frame[:, i:i+20], off, axis=0)

    # ───── sketch ─────
    elif current_filter == "sketch":
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        inv  = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv,(111,111),0)
        sk   = cv2.divide(gray, cv2.bitwise_not(blur), scale=256.0)
        frame[:] = cv2.cvtColor(sk, cv2.COLOR_GRAY2BGR)

    # ───── chath ─────
    elif current_filter == "chath":
        hsv   = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        hue   = int(time.time()*60)%180
        rain  = np.full_like(hsv,(hue,255,255))
        rr    = cv2.cvtColor(rain,cv2.COLOR_HSV2BGR)
        frame[:] = cv2.addWeighted(frame,0.5,rr,0.5,0)
        x0,y0,bw,bh = get_face_box(lm, iw, ih)
        donor = cv2.resize(filters["chath"]["rgb"],(bw,bh),interpolation=cv2.INTER_AREA)
        msk   = cv2.resize(filters["chath"]["mask"],(bw,bh),interpolation=cv2.INTER_NEAREST)
        h_clip = min(bh, ih-y0); w_clip = min(bw, iw-x0)
        donor = donor[:h_clip,:w_clip]; msk = msk[:h_clip,:w_clip]
        roi   = frame[y0:y0+h_clip, x0:x0+w_clip]
        np.copyto(roi, donor, where=msk[...,None].astype(bool))

    # ───── cowboy ─────
    elif current_filter == "cowboy":
        hat = filters["cowboy"]["hat"]
        ms  = filters["cowboy"]["mustache"]
        if hat is not None:
            hw = int((rex-lex)*2.5)+40; hh_ = int(hw*0.6)+20
            hx = fx-hw//2; hy = fy-hh_+10
            if 0<=hx<iw-hw and 0<=hy<ih-hh_:
                r = cv2.resize(hat,(hw,hh_),cv2.INTER_CUBIC)
                frame[hy:hy+hh_, hx:hx+hw] = overlay_image(
                    frame[hy:hy+hh_, hx:hx+hw], r[:,:,:3], r[:,:,3]
                )
        if ms is not None:
            mw = int((rex-lex)*1.2)+40; mh = int(mw*0.3)+20
            mx = nx-mw//2; my = ny-3
            if 0<=mx<iw-mw and 0<=my<ih-mh:
                m_img = cv2.resize(ms,(mw,mh),cv2.INTER_CUBIC)
                frame[my:my+mh, mx:mx+mw] = overlay_image(
                    frame[my:my+mh, mx:mx+mw],
                    m_img[:,:,:3], m_img[:,:,3]
                )

    # ───── cat/dog ─────
    elif current_filter in ("cat","dog"):
        fdata = filters[current_filter]
        # dog tongue
        if current_filter=="dog":
            if (bly - tly) > 15 and "tongue" in fdata:
                tw, th = int(ew*0.6), int(ew*0.6*0.6)
                tx = (lex+rex)//2 - tw//2; ty = ny + int(th*0.6)
                if 0<=tx<iw-tw and 0<=ty<ih-th:
                    t_img = cv2.resize(fdata["tongue"], (tw,th), cv2.INTER_CUBIC)
                    frame[ty:ty+th, tx:tx+tw] = overlay_image(
                        frame[ty:ty+th, tx:tx+tw], t_img[:,:,:3], t_img[:,:,3]
                    )
        # ears
        for part,off in [("left_ear",-ew),("right_ear",0)]:
            if part in fdata:
                ex = fx + off; ey = fy - int(eh*0.9)
                if 0<=ex<iw-ew and 0<=ey<ih-eh:
                    e_img = cv2.resize(fdata[part], (ew,eh), cv2.INTER_CUBIC)
                    frame[ey:ey+eh, ex:ex+ew] = overlay_image(
                        frame[ey:ey+eh, ex:ex+ew],
                        e_img[:,:,:3], e_img[:,:,3]
                    )
        # nose
        if "nose" in fdata:
            nw, nh = int(ew*0.7), int(ew*0.5*0.5)
            nxp = (lex+rex)//2 - nw//2; nyp = ny - nh//2
            if 0<=nxp<iw-nw and 0<=nyp<ih-nh:
                n_img = cv2.resize(fdata["nose"], (nw,nh), cv2.INTER_CUBIC)
                frame[nyp:nyp+nh, nxp:nxp+nw] = overlay_image(
                    frame[nyp:nyp+nh, nxp:nxp+nw],
                    n_img[:,:,:3], n_img[:,:,3]
                )

    # ───── Original ───── (do nothing)

# ─── SERVER API ────────────────────────────────────────────────────────────────
def gen_frames():
    global last_filter
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_mesh.process(rgb)

        # music change
        if current_filter != last_filter:
            pygame.mixer.stop()
            if current_filter == "cowboy": cowboy_music.play(-1)
            if current_filter == "chath":  chath_music.play(-1)
            last_filter = current_filter

        if faces.multi_face_landmarks:
            apply_filter_logic(frame, *frame.shape[1::-1], faces.multi_face_landmarks[0])

        ret2, buf = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )

def set_filter(name):
    global current_filter
    if name in filter_names:
        current_filter = name

# def save_snapshot():
#     ret, frame = cap.read()
#     if not ret: return None
#     frame = cv2.flip(frame, 1)
#     ts    = int(time.time()*1000)
#     fname = f"{current_filter}_{ts}.png"
#     path  = os.path.join(PIC_DIR, fname)
#     cv2.imwrite(path, frame)
#     click.play()
#     return fname

def list_snapshots():
    return sorted(os.listdir(PIC_DIR))

def save_snapshot():
    """
    Capture one frame, apply the active filter,
    save it to disk, and return its filename.
    """
    ret, frame = cap.read()
    if not ret:
        return None

    frame = cv2.flip(frame, 1)
    ih, iw = frame.shape[:2]

    # run face detection
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_mesh.process(rgb)

    # if we got a face, apply stickers+effects
    if faces.multi_face_landmarks:
        apply_filter_logic(frame, iw, ih, faces.multi_face_landmarks[0])
    else:
        # even if no face, apply pure-effects (vintage, mirror, etc.)
        apply_filter_logic(frame, iw, ih, None)

    # save to disk
    ts    = int(time.time() * 1000)
    fname = f"{current_filter}_{ts}.png"
    path  = os.path.join(PIC_DIR, fname)
    cv2.imwrite(path, frame)
    click.play()


return fname