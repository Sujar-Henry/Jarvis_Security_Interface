import cv2
import mediapipe as mp
import time
import numpy as np
import random
import math

# JARVIS UI Drawing Functions
def draw_jarvis_text(frame, text, pos, color=(100, 200, 255), size=1.0, thickness=2):
    for i in range(2):
        glow_color = (color[0]//5, color[1]//5, color[2]//5)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, 
                   glow_color, thickness + i*2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

def draw_jarvis_circle(frame, center, radius, color=(100, 200, 255), thickness=2):
    cv2.circle(frame, center, radius, color, thickness)
    cv2.circle(frame, center, radius-5, (color[0]//2, color[1]//2, color[2]//2), 1)

def draw_arc_scanner(frame, center, radius, angle_start, angle_end, color=(100, 200, 255)):
    pts = []
    for angle in np.linspace(angle_start, angle_end, 20):
        x = center[0] + int(radius * np.cos(angle))
        y = center[1] + int(radius * np.sin(angle))
        pts.append([x, y])
    
    if len(pts) > 1:
        pts = np.array(pts, np.int32)
        cv2.polylines(frame, [pts], False, color, 3)

def draw_jarvis_hud_box(frame, x, y, w, h, color=(100, 200, 255)):
    corner_size = 25
    thickness = 2
    
    cv2.rectangle(frame, (x, y), (x+w, y+h), (color[0]//3, color[1]//3, color[2]//3), 1)
    
    corners = [
        [(x, y), (x + corner_size, y), (x, y + corner_size)],
        [(x + w, y), (x + w - corner_size, y), (x + w, y + corner_size)],
        [(x, y + h), (x + corner_size, y + h), (x, y + h - corner_size)],
        [(x + w, y + h), (x + w - corner_size, y + h), (x + w, y + h - corner_size)]
    ]
    
    for corner in corners:
        cv2.line(frame, corner[0], corner[1], color, thickness)
        cv2.line(frame, corner[0], corner[2], color, thickness)

def draw_jarvis_overlay(frame):
    h, w = frame.shape[:2]
    
    bracket_size = 36
    color = (100, 200, 255)
    thickness = 2
    margin = 18
    cv2.line(frame, (margin, margin), (margin + bracket_size, margin), color, thickness)
    cv2.line(frame, (margin, margin), (margin, margin + bracket_size), color, thickness)
    cv2.line(frame, (w-margin, margin), (w-margin - bracket_size, margin), color, thickness)
    cv2.line(frame, (w-margin, margin), (w-margin, margin + bracket_size), color, thickness)
    cv2.line(frame, (margin, h-margin), (margin + bracket_size, h-margin), color, thickness)
    cv2.line(frame, (margin, h-margin), (margin, h-margin - bracket_size), color, thickness)
    cv2.line(frame, (w-margin, h-margin), (w-margin - bracket_size, h-margin), color, thickness)
    cv2.line(frame, (w-margin, h-margin), (w-margin, h-margin - bracket_size), color, thickness)

def draw_hexagon(frame, center, radius, color=(100, 200, 255), thickness=2):
    points = []
    for i in range(6):
        angle = i * np.pi / 3
        x = center[0] + int(radius * np.cos(angle))
        y = center[1] + int(radius * np.sin(angle))
        points.append([x, y])
    
    points = np.array(points, np.int32)
    cv2.polylines(frame, [points], True, color, thickness)

# Initialize MediaPipe and OpenCV components
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
AUTHORIZED_FACE_IMG = 'authorized_face.jpg'

try:
    auth_img = cv2.imread(AUTHORIZED_FACE_IMG, cv2.IMREAD_GRAYSCALE)
    if auth_img is not None:
        auth_hist = cv2.calcHist([auth_img], [0], None, [256], [0, 256])
        print("▓ JARVIS: Authorized biometric profile loaded")
    else:
        raise FileNotFoundError
except:
    print("▓ JARVIS: Authorized profile not found. Initiating demo protocol")
    auth_img = None
    auth_hist = None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                      min_detection_confidence=0.7, min_tracking_confidence=0.7)

# System configuration
FINGER_PASSWORD = [2, 3, 1, 0]
BUTTON_CENTER = (320, 400)
BUTTON_RADIUS = 80

# State variables
mode = 'startup'
startup_time = time.time()
password_progress = 0
last_finger_time = 0
unlocked = False
scan_angle = 0

cap = cv2.VideoCapture(0)
print('▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓')
print('▓ JARVIS NEURAL SECURITY INTERFACE V3.2 ▓')
print('▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓')
print('▓ JARVIS: Good day, Sir. Initializing security protocols...')

# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    blue_overlay = np.zeros_like(frame)
    blue_overlay[:, :] = (12, 8, 0)
    frame = cv2.addWeighted(frame, 0.92, blue_overlay, 0.08, 0)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    draw_jarvis_overlay(frame)

    # State machine for different interface modes
    if mode == 'startup':
        startup_progress = time.time() - startup_time
        
        if startup_progress < 2.5:
            draw_jarvis_text(frame, 'JARVIS', (w//2-70, h//2-40), size=1.7, thickness=4)
            draw_jarvis_text(frame, 'Neural Security Interface', (w//2-150, h//2+15), size=0.7)
            radius = 45 + int(18 * abs(np.sin(startup_progress * 2)))
            color = (60, 120, 255)
            draw_jarvis_circle(frame, (w//2, h//2+90), radius, color, 2)
        else:
            mode = 'initialize'

    elif mode == 'initialize':
        draw_jarvis_text(frame, 'JARVIS SECURITY PROTOCOL', (w//2-170, 80), size=0.9, thickness=3)
        pulse = int(12 * abs(np.sin(time.time() * 2)))
        draw_hexagon(frame, BUTTON_CENTER, BUTTON_RADIUS + pulse, (100, 200, 255), 2)
        draw_hexagon(frame, BUTTON_CENTER, BUTTON_RADIUS, (150, 220, 255), 1)
        cv2.circle(frame, BUTTON_CENTER, BUTTON_RADIUS-18, (50, 100, 150), -1)
        draw_jarvis_text(frame, 'ACTIVATE', (BUTTON_CENTER[0]-48, BUTTON_CENTER[1]+6), size=0.7)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            idx_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            idx_xy = (int(idx_tip.x * w), int(idx_tip.y * h))
            
            cv2.circle(frame, idx_xy, 12, (100, 200, 255), 3)
            cv2.circle(frame, idx_xy, 6, (255, 255, 255), -1)
            
            if np.linalg.norm(np.array(idx_xy) - np.array(BUTTON_CENTER)) < BUTTON_RADIUS:
                mode = 'face'
                face_scan_start = time.time()
                continue

    elif mode == 'face':
        draw_jarvis_text(frame, 'BIOMETRIC SCAN ACTIVE', (w//2-140, 60), size=0.9, thickness=3)
        
        scan_angle += 0.1
        center = (w//2, h//2)
        for i in range(4):
            angle_start = scan_angle + i * np.pi/2
            angle_end = angle_start + np.pi/4
            radius = 200 + i * 30
            draw_arc_scanner(frame, center, radius, angle_start, angle_end, (100, 200, 255))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        draw_jarvis_text(frame, 'Analyzing facial biometric patterns...', (w//2-150, h-80), size=0.6)
        grid_speed = 2.5
        grid_offset = int(30 * np.sin(time.time() * grid_speed))
        grid_alpha = int(50 * abs(np.sin(time.time() * 3)))
        if len(faces) > 0:
            for (x, y, fw, fh) in faces:
                for gx in range(x, x+fw, 24):
                    cv2.line(frame, (gx, y), (gx, y+fh), (0, grid_alpha, grid_alpha*2), 1)
                for gy in range(y + grid_offset, y+fh, 24):
                    cv2.line(frame, (x, gy), (x+fw, gy), (0, grid_alpha, grid_alpha*2), 1)
        else:
            cx, cy = w//2, h//2
            for gx in range(cx-80, cx+80, 24):
                cv2.line(frame, (gx, cy-80), (gx, cy+80), (0, grid_alpha, grid_alpha*2), 1)
            for gy in range(cy-80 + grid_offset, cy+80, 24):
                cv2.line(frame, (cx-80, gy), (cx+80, gy), (0, grid_alpha, grid_alpha*2), 1)
        
        for (x, y, fw, fh) in faces:
            draw_jarvis_hud_box(frame, x-15, y-15, fw+30, fh+30, (100, 200, 255))
            
            center_x, center_y = x + fw//2, y + fh//2
            draw_jarvis_circle(frame, (center_x, center_y), 40, (100, 200, 255), 2)
            cv2.line(frame, (center_x-50, center_y), (center_x+50, center_y), (100, 200, 255), 2)
            cv2.line(frame, (center_x, center_y-50), (center_x, center_y+50), (100, 200, 255), 2)
            
            face_roi = gray[y:y + fh, x:x + fw]
            scan_time = time.time() - face_scan_start
            
            if auth_img is not None and auth_hist is not None:
                face_resized = cv2.resize(face_roi, (auth_img.shape[1], auth_img.shape[0]))
                face_hist = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
                similarity = cv2.compareHist(auth_hist, face_hist, cv2.HISTCMP_CORREL)
                
                draw_jarvis_text(frame, f'BIOMETRIC MATCH: {similarity:.3f}', (x, y-25), size=0.6)
                
                if similarity > 0.8 and scan_time > 3:
                    mode = 'password'
                    password_progress = 0
                    last_finger_time = 0
                    print("▓ JARVIS: Biometric authentication successful")
                    continue
            else:
                demo_progress = min(0.999, scan_time / 4.0)
                draw_jarvis_text(frame, f'BIOMETRIC MATCH: {demo_progress:.3f}', (x, y-25), size=0.6)
        
        if time.time() - face_scan_start > 6:
            mode = 'password'
            password_progress = 0
            last_finger_time = 0
            print("▓ JARVIS: Proceeding to neural interface authentication")

    elif mode == 'password':
        draw_jarvis_text(frame, 'NEURAL INTERFACE AUTHENTICATION', (w//2-180, 60), size=0.9, thickness=3)
        draw_jarvis_text(frame, 'Input gesture sequence', (w//2-90, 100), size=0.6)
        
        progress_center = (w//2, h//2 + 30)
        progress_radius = 60
        draw_jarvis_circle(frame, progress_center, progress_radius, (100, 200, 255), 3)

        if password_progress > 0:
            progress_angle = (password_progress / len(FINGER_PASSWORD)) * 2 * np.pi
            draw_arc_scanner(frame, progress_center, progress_radius-5, 0, progress_angle, (150, 255, 150))

        sequence_y = h//2 + 80
        for i in range(len(FINGER_PASSWORD)):
            x_pos = w//2 - 70 + i * 40
            if i < password_progress:
                color = (100, 255, 100)
                status = "✓"
            else:
                color = (80, 80, 80)
                status = "○"
            draw_jarvis_text(frame, status, (x_pos, sequence_y), color, 0.8)

        finger_count = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
            
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(100,200,255), thickness=2, circle_radius=3),
                mp.solutions.drawing_utils.DrawingSpec(color=(150,220,255), thickness=2)
            )
            
            hand_label = None
            if hasattr(results, 'multi_handedness') and results.multi_handedness:
                hand_label = results.multi_handedness[0].classification[0].label
            
            tips = [4, 8, 12, 16, 20]
            fingers = []
            
            if hand_label == 'Right':
                fingers.append(lm[tips[0]].x > lm[tips[0] - 1].x)
            else:
                fingers.append(lm[tips[0]].x < lm[tips[0] - 1].x)
                
            for i in range(1, 5):
                fingers.append(lm[tips[i]].y < lm[tips[i] - 2].y)
                
            finger_count = sum(fingers)
            
            draw_jarvis_text(frame, f'INPUT: {finger_count}', (30, h - 70), size=0.7)
            draw_jarvis_text(frame, f'EXPECT: {FINGER_PASSWORD[password_progress] if password_progress < len(FINGER_PASSWORD) else "DONE"}', 
                          (30, h - 40), size=0.6)

        now = time.time()
        FINGER_ENTRY_DELAY = 1.5
        
        if not unlocked and password_progress < len(FINGER_PASSWORD):
            if finger_count == FINGER_PASSWORD[password_progress]:
                if now - last_finger_time > FINGER_ENTRY_DELAY:
                    password_progress += 1
                    last_finger_time = now
                    print(f"▓ JARVIS: Gesture {password_progress} verified")
                    if password_progress == len(FINGER_PASSWORD):
                        unlocked = True
                        unlock_time = time.time()
                        print("▓ JARVIS: Neural interface authentication complete")
            elif finger_count is not None and finger_count != FINGER_PASSWORD[password_progress]:
                if now - last_finger_time > FINGER_ENTRY_DELAY:
                    password_progress = 0
                    last_finger_time = now
                    print("▓ JARVIS: Incorrect gesture. Sequence reset.")
        
        if unlocked:
            grant_color = (0, 255, 180)
            draw_jarvis_text(frame, 'ACCESS GRANTED', (w//2-150, h-30), grant_color, 1.2, 3)
            

    draw_jarvis_text(frame, f'STATUS: {mode.upper()}', (24, 38), size=0.5)
    draw_jarvis_text(frame, 'JARVIS v3.2', (24, h - 18), size=0.45)
    draw_jarvis_text(frame, 'Q: Disconnect', (w - 140, h - 18), size=0.45)

    cv2.imshow('JARVIS Neural Security Interface', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('▓ JARVIS: Neural security interface disconnected. Have a good day, Sir.')