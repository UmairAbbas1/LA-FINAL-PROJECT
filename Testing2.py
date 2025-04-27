import cv2
import numpy as np
import os

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = r"D:\LA-FINAL-PROJECT"
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Face detector (SSD Caffe)
FACE_PROTO  = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_MODEL  = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Gender net
GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")

# Age net
AGE_PROTO    = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL    = os.path.join(MODEL_DIR, "age_net.caffemodel")

# Mean values for the models
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Age buckets and genders
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
               "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDERS = ["Male", "Female"]

# -----------------------------------------------------------------------------
# LOAD NETWORKS
# -----------------------------------------------------------------------------
# Face detector
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
# Gender classifier
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
# Age classifier
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

# -----------------------------------------------------------------------------
# CAMERA SETUP
# -----------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open webcam")

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Detect faces
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104.0, 177.0, 123.0], swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.6:
            continue

        # Compute bounding box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Extract the face ROI
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Prepare blob for gender/age nets
        face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                          MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        gender_net.setInput(face_blob)
        gender_preds = gender_net.forward()
        gender = GENDERS[gender_preds[0].argmax()]

        # Predict age
        age_net.setInput(face_blob)
        age_preds = age_net.forward()
        age = AGE_BUCKETS[age_preds[0].argmax()]

        # Overlay results
        label = f"{gender}, {age}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show
    cv2.imshow("Age & Gender Test", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
