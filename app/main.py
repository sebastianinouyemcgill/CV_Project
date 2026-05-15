import cv2
from db import init_db, log_attendance
import math

# Initialize DB
init_db()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

print("Press 'q' to quit. Type name in terminal when prompted for new faces.")

# Track faces: key=face_id, value=(cx, cy, w, h, name, last_seen_frame)
known_faces = {}
attendance_today = set()
DISTANCE_THRESHOLD = 80    # pixels
FRAME_TTL = 30             # frames to consider a face “still present”

frame_count = 0

def get_centroid(box):
    x, y, w, h = box
    return x + w // 2, y + h // 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Mark all known faces as unseen this frame
    for face_id in list(known_faces.keys()):
        known_faces[face_id]['seen'] = False

    for (x, y, w, h) in faces:
        cx, cy = get_centroid((x, y, w, h))
        matched = None

        # Match against existing faces
        for fid, data in known_faces.items():
            fx, fy, fw, fh, name, last_seen, _ = data.values()
            dist = math.hypot(cx - fx, cy - fy)
            if dist < DISTANCE_THRESHOLD:
                matched = fid
                break

        if matched:
            known_faces[matched]['cx'] = cx
            known_faces[matched]['cy'] = cy
            known_faces[matched]['w'] = w
            known_faces[matched]['h'] = h
            known_faces[matched]['last_seen'] = frame_count
            known_faces[matched]['seen'] = True
            display_name = known_faces[matched]['name']
        else:
            # New face
            display_name = input("New face detected! Enter your name: ")
            known_faces[frame_count] = {
                'cx': cx, 'cy': cy, 'w': w, 'h': h,
                'name': display_name, 'last_seen': frame_count, 'seen': True
            }
            if display_name not in attendance_today:
                log_attendance(display_name)
                attendance_today.add(display_name)

        # Draw box and name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, display_name, (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Remove old faces not seen for FRAME_TTL frames
    to_delete = [fid for fid, data in known_faces.items() if frame_count - data['last_seen'] > FRAME_TTL]
    for fid in to_delete:
        del known_faces[fid]

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()