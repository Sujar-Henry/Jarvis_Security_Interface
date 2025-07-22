# 🧠 JARVIS Interface(Computer Vision)

A futuristic biometric security interface inspired by JARVIS (Iron Man) using computer vision. This Python project combines facial recognition and hand gesture password authentication with a sleek, animated sci-fi heads-up display (HUD) built using OpenCV and MediaPipe.

---

## 🔐 Features

- **Three-Stage Authentication**  
  1. **Startup Sequence**: Animated JARVIS boot UI.  
  2. **Biometric Scan**: Facial recognition using OpenCV Haar cascades.  
  3. **Gesture Password**: Finger gesture sequence detection using MediaPipe Hands.

- **Sci-Fi Visual Interface**  
  - JARVIS-style HUD elements: hexagons, glowing text, scanners, and progress arcs.
  - Smooth animations and progress indicators.
  - Dynamic biometric similarity match %.
  - Hand gesture recognition with custom password logic.

- **Demo Mode Fallback**  
  If the authorized face image is missing, the interface still runs using realistic simulated progress and animations.

---

## 📸 Requirements

- Python 3.7+
- Webcam

---

## 🧰 Libraries

```bash
pip install opencv-python mediapipe numpy


├── jarvis_security_interface.py     # Main script
├── authorized_face.jpg              # Reference image for biometric scan (optional)

📌 Note: Place a  facial image named authorized_face.jpg in the root folder for biometric authentication. Without it, the system runs in demo mode.

## How to Run

```python
python jarvis_security_interface.py
```


## Manual Set password

```
[2, 3, 1, 0] → Peace ✌️, Three 🖖, Index ☝️, Fist ✊

```