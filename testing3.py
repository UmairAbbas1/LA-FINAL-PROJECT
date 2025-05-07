import cv2
import time

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Draw green box around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show frame
    cv2.imshow('Face Detection', frame)

    # Save image on 's' key press
    key = cv2.waitKey(1)
    if key == ord('s'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f'face_capture_{timestamp}.jpg', frame)
    # Quit on 'q'
    elif key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows() 