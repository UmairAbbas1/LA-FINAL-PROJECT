# Face Filter App – FULL VERSION with fixed beauty/brighten filters
# ------------------------------------------------------------------------
# Requirements:
#   pip install opencv-python mediapipe pygame numpy
# Assets expected in D:\LA-FINAL-PROJECT\:
#   ├─ Beep.wav
#   ├─ iphone-camera-capture-6448.wav
#   ├─ cowboy_music.wav
#   ├─ CHATH.png
#   ├─ BADO BADI.wav
#   ├─ left.png, right.png, nose.png
#   ├─ Dogleft.png, Dogright.png, Dognose.png, tongue.png
#   └─ Pictures\  (auto‑created for snapshots)
# ------------------------------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time
import random

################################################################################
#                               INITIAL SETUP                                  #
################################################################################

BASE_DIR = r"D:\LA-FINAL-PROJECT"
PIC_DIR  = os.path.join(BASE_DIR, "Pictures")
os.makedirs(PIC_DIR, exist_ok=True)

# Pygame for audio
pygame.init()
beep          = pygame.mixer.Sound(os.path.join(BASE_DIR, "Beep.wav"))
click         = pygame.mixer.Sound(os.path.join(BASE_DIR, "iphone-camera-capture-6448.wav"))
cowboy_music  = pygame.mixer.Sound(os.path.join(BASE_DIR, "cowboy_music.wav"))
chath_music   = pygame.mixer.Sound(os.path.join(BASE_DIR, "BADO BADI.wav"))

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("[ERROR] Unable to open webcam.")

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands     = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands     = mp_hands.Hands(max_num_hands=1)

################################################################################
#                           LOAD FILTER IMAGES                                 #
################################################################################

def png(path):
    return cv2.imread(os.path.join(BASE_DIR, path), cv2.IMREAD_UNCHANGED)

filters = {
    # sticker‑based filters
    "cat": {
        "left_ear" : png("left.png"),
        "right_ear": png("right.png"),
        "nose"     : png("nose.png"),
    },
    "dog": {
        "left_ear" : png("Dogleft.png"),
        "right_ear": png("Dogright.png"),
        "nose"     : png("Dognose.png"),
        "tongue"   : png("tongue.png"),
    },
    "glasses": {
        "glass": png("Glasses.png"),
    },
    "cowboy": {
        "hat"     : png("cowboy_hat.png"),
        "mustache": png("moustache.png"),
    },
    "chath": {},  # filled below
    # effect filters
    "vintage"      : {},
    "brighten"     : {},
    "Black & white": {},
    "cartoon"      : {},
    "beauty"       : {},
    "sketch"       : {},
    "negative"     : {},
    "sepia"        : {},
    "glitch"       : {},
    "neon_glow"    : {},
    "Original"     : {},
}

# Prepare CHATH donor face & mask
CHATH_PNG = png("CHATH.png")
if CHATH_PNG is None or CHATH_PNG.shape[2] != 4:
    raise FileNotFoundError("CHATH.png missing or lacks alpha channel!")
alpha = CHATH_PNG[:, :, 3]
mask_ch = (alpha > 200).astype(np.uint8) * 255
contours, _ = cv2.findContours(mask_ch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise RuntimeError("No contour found in CHATH.png!")
xx, yy, ww, hh = cv2.boundingRect(contours[0])
chath_rgb  = CHATH_PNG[yy:yy+hh, xx:xx+ww, :3]
chath_mask = mask_ch[yy:yy+hh, xx:xx+ww]
filters["chath"] = {"rgb": chath_rgb, "mask": chath_mask}

# UI panel list
filter_names = list(filters.keys())

################################################################################
#                   GLOBAL STATE FOR PANEL & SNAPSHOT LOGIC                    #
################################################################################

active_filter   = filter_names[0]
current_filter  = active_filter
last_filter     = None

panel_open      = False
panel_x, panel_y = 10, 60
panel_drag       = False
panel_width      = 160
panel_h_vis      = 300
bar_h            = 20
arrow_h          = 30
btn_h            = 40
pad              = 10
SCROLL_STEP      = btn_h + pad
scroll_offset    = 0
MIN_SCROLL, MAX_SCROLL = 0, 0

timer_started   = False
countdown_start = 0
cooldown        = False
cooldown_start  = 0
COOLDOWN_TIME   = 5
COUNTDOWN_TIME  = 3
last_beep_time  = -1
pic_count       = 0

################################################################################
#                            UTILITY FUNCTIONS                                 #
################################################################################

def overlay_image(roi, part_img, part_mask):
    roi = roi.astype(np.uint8)
    part_img = part_img.astype(np.uint8)
    part_mask = part_mask.astype(np.uint8)
    if roi.shape[:2] != part_img.shape[:2]:
        part_img  = cv2.resize(part_img, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)
        part_mask = cv2.resize(part_mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    inv = cv2.bitwise_not(part_mask)
    bg = cv2.bitwise_and(roi, roi, mask=inv)
    fg = cv2.bitwise_and(part_img, part_img, mask=part_mask)
    return cv2.add(bg, fg)

def save_picture(frame):
    global pic_count
    fname = os.path.join(PIC_DIR, f"{active_filter}_filter_{pic_count}.png")
    cv2.imwrite(fname, frame)
    print(f"[✅] Picture saved as {fname}")
    click.play()
    pic_count += 1

def get_face_box(face_landmarks, w, h):
    """
    Compute bounding box around key facial landmarks.
    """
    pts = np.array([[p.x*w, p.y*h] for p in face_landmarks.landmark])
    idx = [10, 152, 234, 454]
    hull = pts[idx]
    x, y, bw, bh = cv2.boundingRect(hull.astype(np.float32))
    pad_w, pad_h = int(bw*0.3), int(bh*0.4)
    x0 = max(0, x-pad_w); y0 = max(0, y-pad_h)
    x1 = min(w, x+bw+pad_w); y1 = min(h, y+bh)
    return x0, y0, x1-x0, y1-y0

################################################################################
#                           UI DRAWING FUNCTIONS                               #
################################################################################

def draw_toggle(frame):
    x,y,wb,hb = 10,10,30,30
    cv2.rectangle(frame,(x,y),(x+wb,y+hb),(0,255,0),-1)
    arrow = "<" if panel_open else ">"
    cv2.putText(frame,arrow,(x+7,y+22),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

def draw_panel(frame):
    global scroll_offset, MIN_SCROLL, MAX_SCROLL
    if not panel_open: return
    t1,t2 = panel_y-bar_h, panel_y
    b = panel_y+panel_h_vis
    cv2.rectangle(frame,(panel_x,t1),(panel_x+panel_width,b),(50,50,50),2)
    cv2.rectangle(frame,(panel_x,t1),(panel_x+panel_width,t2),(100,100,100),-1)
    cv2.putText(frame,"Move",(panel_x+10,panel_y-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
    up1,up2 = panel_y, panel_y+arrow_h
    cv2.rectangle(frame,(panel_x,up1),(panel_x+panel_width,up2),(120,120,120),-1)
    cv2.putText(frame,"Up",(panel_x+60,up2-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    down2,down1 = b, b-arrow_h
    cv2.rectangle(frame,(panel_x,down1),(panel_x+panel_width,down2),(120,120,120),-1)
    cv2.putText(frame,"Down",(panel_x+45,down2-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    at,ab = up2, down1
    scroll_h = ab - at
    tot_h = len(filter_names)*(btn_h+pad)-pad
    over  = max(0, tot_h-scroll_h)
    MIN_SCROLL, MAX_SCROLL = -over, 0
    scroll_offset = max(MIN_SCROLL, min(MAX_SCROLL, scroll_offset))
    y0 = at + scroll_offset
    for i,name in enumerate(filter_names):
        y = y0 + i*(btn_h+pad)
        if y+btn_h<at or y>ab-btn_h: continue
        col = (0,255,255) if name==active_filter else (200,200,200)
        cv2.rectangle(frame,(panel_x,y),(panel_x+panel_width,y+btn_h),col,-1)
        cv2.putText(frame,name[:12],(panel_x+5,y+25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1)

################################################################################
#                         MOUSE CALLBACK (UI)                                  #
################################################################################

def mouse_cb(evt,x,y,flags,param):
    global panel_open,panel_drag,panel_x,panel_y,scroll_offset
    global active_filter,current_filter
    if evt==cv2.EVENT_LBUTTONDOWN:
        if 10<=x<=40 and 10<=y<=40:
            panel_open=not panel_open; return
        if panel_open:
            t1,t2=panel_y-bar_h,panel_y
            up1,up2=panel_y,panel_y+arrow_h
            b=panel_y+panel_h_vis; d1=b-arrow_h
            if panel_x<=x<=panel_x+panel_width and t1<=y<=b:
                if t1<=y<=t2:
                    panel_drag=True; return
                if up1<=y<=up2:
                    scroll_offset+=SCROLL_STEP; return
                if d1<=y<=b:
                    scroll_offset-=SCROLL_STEP; return
                top,bot=up2,d1; y0=top+scroll_offset
                for i,nm in enumerate(filter_names):
                    yy=y0+i*(btn_h+pad)
                    if yy<=y<=yy+btn_h and panel_x<=x<=panel_x+panel_width:
                        active_filter=current_filter=nm; return
    elif evt==cv2.EVENT_LBUTTONUP:
        panel_drag=False
    elif evt==cv2.EVENT_MOUSEMOVE and panel_drag:
        panel_x=x-panel_width//2
        panel_y=max(40,y)

################################################################################
#                        FILTER APPLICATION LOGIC                              #
################################################################################

def apply_filter_logic(frame, iw, ih, landmarks):
    global current_filter
    le, re = landmarks.landmark[33], landmarks.landmark[263]
    nt, fh = landmarks.landmark[1], landmarks.landmark[10]
    tl, bl = landmarks.landmark[13], landmarks.landmark[14]
    lex,ley = int(le.x*iw), int(le.y*ih)
    rex,rey = int(re.x*iw), int(re.y*ih)
    nx,ny   = int(nt.x*iw), int(nt.y*ih)
    fx,fy   = int(fh.x*iw), int(fh.y*ih)
    tly,bly = int(tl.y*ih), int(bl.y*ih)
    ew = int((rex-lex)*1.8); eh=int(ew*0.75)
    fdata = filters[current_filter]

    # vintage
    if current_filter=="vintage":
        sep=cv2.transform(frame,np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]]))
        frame[:]=np.clip(sep,0,255).astype(np.uint8)

    # glasses
    elif current_filter=="glasses":
        if "glass" in fdata:
            g=fdata["glass"]; gw=int((rex-lex)*2); gh=int(gw*0.4)
            r=cv2.resize(g,(gw,gh),cv2.INTER_CUBIC)
            rgb,msk=r[:,:,:3],r[:,:,3]
            gx,gy=lex-int(gw*0.25),ley-gh//2
            if 0<=gx<iw-gw and 0<=gy<ih-gh:
                frame[gy:gy+gh,gx:gx+gw]=overlay_image(frame[gy:gy+gh,gx:gx+gw],rgb,msk)

    # neon_glow
    elif current_filter=="neon_glow":
        gr=cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(21,21),0)
        _,e=cv2.threshold(gr,50,255,cv2.THRESH_BINARY)
        e=cv2.cvtColor(e,cv2.COLOR_GRAY2BGR)
        frame[:]=cv2.addWeighted(frame,0.7,e,0.3,0)

    # Black & white
    elif current_filter=="Black & white":
        g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame[:]=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)

    # cartoon
    elif current_filter=="cartoon":
        c=frame.copy()
        for _ in range(3): c=cv2.bilateralFilter(c,9,10,40)
        g=cv2.medianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),7)
        e=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,2)
        e=cv2.cvtColor(e,cv2.COLOR_GRAY2BGR)
        c2=cv2.bitwise_and(c,e)
        frame[:]=cv2.filter2D(c2,-1,np.array([[-1,-1,-1],[-1,30,-1],[-1,-1,-1]]))

    # beauty (fixed)
    elif current_filter=="beauty":
        bil=cv2.bilateralFilter(frame,d=15,sigmaColor=75,sigmaSpace=75)
        frame[:]=cv2.addWeighted(bil,0.6,frame,0.4,0)

    # brighten (fixed)
    elif current_filter=="brighten":
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv)
        v=np.clip(v+30,0,255).astype(np.uint8)
        frame[:]=cv2.cvtColor(cv2.merge((h,s,v)),cv2.COLOR_HSV2BGR)

    # negative
    elif current_filter=="negative":
        frame[:]=cv2.bitwise_not(frame)

    # sepia
    elif current_filter=="sepia":
        t=cv2.transform(frame,np.array([[0.393,0.769,0.189],[0.349,0.686,0.168],[0.272,0.534,0.131]]))
        frame[:]=np.clip(t,0,255).astype(np.uint8)

    # glitch
    elif current_filter=="glitch":
        hh,ww,_=frame.shape
        for i in range(0,ww,random.randint(20,100)):
            off=random.randint(-10,10)
            frame[:,i:i+20]=np.roll(frame[:,i:i+20],off,axis=0)

    # sketch
    elif current_filter=="sketch":
        g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        inv=cv2.bitwise_not(g)
        blur=cv2.GaussianBlur(inv,(111,111),0)
        sk=cv2.divide(g,cv2.bitwise_not(blur),scale=256.0)
        frame[:]=cv2.cvtColor(sk,cv2.COLOR_GRAY2BGR)

    # chath
    elif current_filter=="chath":
        # rainbow tint
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        hue=(int(time.time()*60)%180)
        rain=np.full_like(hsv,(hue,255,255))
        rr=cv2.cvtColor(rain,cv2.COLOR_HSV2BGR)
        frame[:]=cv2.addWeighted(frame,0.5,rr,0.5,0)
        # overlay face
        x0,y0,bw,bh=get_face_box(landmarks,iw,ih)
        donor=cv2.resize(filters["chath"]["rgb"],(bw,bh),cv2.INTER_AREA)
        msk=cv2.resize(filters["chath"]["mask"],(bw,bh),cv2.INTER_NEAREST)
        h_clip=min(bh,ih-y0); w_clip=min(bw,iw-x0)
        donor=donor[:h_clip,:w_clip]; msk=msk[:h_clip,:w_clip]
        roi=frame[y0:y0+h_clip,x0:x0+w_clip]
        np.copyto(roi,donor,where=msk[...,None].astype(bool))

    # cowboy
    elif current_filter=="cowboy":
        hat=fdata.get("hat"); ms=fdata.get("mustache")
        if hat is not None:
            hw=int((rex-lex)*2.5)+40; hh=int(hw*0.6)+20
            hx,hy=fx-hw//2,fy-hh+10
            if 0<=hx<iw-hw and 0<=hy<ih-hh:
                r=cv2.resize(hat,(hw,hh),cv2.INTER_CUBIC)
                frame[hy:hy+hh,hx:hx+hw]=overlay_image(frame[hy:hy+hh,hx:hx+hw],r[:,:,:3],r[:,:,3])
        if ms is not None:
            mw=int((rex-lex)*1.2)+40; mh=int(mw*0.3)+20
            mx,my=nx-mw//2,ny-3
            if 0<=mx<iw-mw and 0<=my<ih-mh:
                m=cv2.resize(ms,(mw,mh),cv2.INTER_CUBIC)
                frame[my:my+mh,mx:mx+mw]=overlay_image(frame[my:my+mh,mx:mx+mw],m[:,:,:3],m[:,:,3])

    # cat/dog
    elif current_filter in ("cat","dog"):
        if current_filter=="dog":
            if (bly-tly)>15 and "tongue" in fdata:
                tw=int(ew*0.6); th=int(tw*0.6)
                tx=(lex+rex)//2-tw//2; ty=ny+int(th*0.6)
                if 0<=tx<iw-tw and 0<=ty<ih-th:
                    timg=cv2.resize(fdata["tongue"],(tw,th),cv2.INTER_CUBIC)
                    frame[ty:ty+th,tx:tx+tw]=overlay_image(frame[ty:ty+th,tx:tx+tw],timg[:,:,:3],timg[:,:,3])
        for part,off in [("left_ear",-ew),("right_ear",0)]:
            if part in fdata:
                ex,ey=fx+off,fy-int(eh*0.9)
                if 0<=ex<iw-ew and 0<=ey<ih-eh:
                    eimg=cv2.resize(fdata[part],(ew,eh),cv2.INTER_CUBIC)
                    frame[ey:ey+eh,ex:ex+ew]=overlay_image(frame[ey:ey+eh,ex:ex+ew],eimg[:,:,:3],eimg[:,:,3])
        if "nose" in fdata:
            nw=int(ew*0.7); nh=int(nw*0.5)
            nxp, nyp=(lex+rex)//2-nw//2,ny-nh//2
            if 0<=nxp<iw-nw and 0<=nyp<ih-nh:
                nimg=cv2.resize(fdata["nose"],(nw,nh),cv2.INTER_CUBIC)
                frame[nyp:nyp+nh,nxp:nxp+nw]=overlay_image(frame[nyp:nyp+nh,nxp:nxp+nw],nimg[:,:,:3],nimg[:,:,3])

    # Original: do nothing

################################################################################
#                                  MAIN LOOP                                   #
################################################################################

cv2.namedWindow("Face Filter App 😺🐶🤓")
cv2.setMouseCallback("Face Filter App 😺🐶🤓", mouse_cb)

while True:
    ret, frame = cap.read()
    if not ret:
        frame = np.zeros((480,640,3),dtype=np.uint8)
    frame = cv2.flip(frame,1)

    rgb       = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces     = face_mesh.process(rgb)
    hands_res = hands.process(rgb)
    ih, iw,_  = frame.shape

    # music switch
    if current_filter!=last_filter:
        pygame.mixer.stop()
        if current_filter=="cowboy":
            cowboy_music.play(-1)
        elif current_filter=="chath":
            chath_music.play(-1)
        last_filter=current_filter

    # hand raise
    raised=False
    if hands_res.multi_hand_landmarks:
        for hl in hands_res.multi_hand_landmarks:
            if hl.landmark[8].y*ih < hl.landmark[0].y*ih:
                raised=True

    # apply filter
    if faces.multi_face_landmarks:
        apply_filter_logic(frame, iw, ih, faces.multi_face_landmarks[0])

    # countdown & snapshot
    now=time.time()
    if raised and not timer_started and not cooldown:
        timer_started=True; countdown_start=now; last_beep_time=-1
    if timer_started:
        el=int(now-countdown_start)
        tl=COUNTDOWN_TIME-el
        if tl>0:
            cv2.putText(frame,str(tl),(50,100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5)
            if el!=last_beep_time:
                if not pygame.mixer.get_busy():
                    beep.play()
                last_beep_time=el
        else:
            save_picture(frame)
            timer_started=False; cooldown=True; cooldown_start=now
    if cooldown and (now-cooldown_start>COOLDOWN_TIME):
        cooldown=False

    # UI & display
    draw_panel(frame)
    draw_toggle(frame)

    cv2.imshow("Face Filter App 😺🐶🤓", frame)
    k=cv2.waitKey(1)&0xFF
    if k==ord('q'):
        break
    if k==ord('s'):
        save_picture(frame)

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
