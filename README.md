# 🖐️ Finger Calculator

A Python-based computer vision application that lets you perform arithmetic calculations using hand gestures! Uses OpenCV and MediaPipe for real-time hand tracking.

---

## ✨ Features

- 🔢 Show numbers using your fingers (0–10 using both hands)
- ➕ Supports all 4 basic operations: Addition, Subtraction, Multiplication, Division
- 🤚 Real-time hand gesture detection using MediaPipe
- ⏱️ Stable hold-to-confirm system (hold fingers still for 3 seconds)
- 🔵 Step-by-step visual UI with live equation display
- ⏳ Animated countdown ring and progress bar
- ❌ Division by zero error handling
- 🎯 FPS counter for performance monitoring
- 🔄 Press `r` to restart, `q` to quit

---

## 🖥️ Demo / Screenshot

![Finger Calculator Screenshot](screenshot.png)

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.x | Core language |
| OpenCV (`cv2`) | Camera feed & drawing UI |
| MediaPipe | Real-time hand landmark detection |

---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/Md-Kaif534/Finger-Calculator.git
cd Finger
```

2. **Install dependencies**
```bash
pip install opencv-python mediapipe
```

3. **Run the project**
```bash
python finger.py
```

---

## 🎮 How to Use

| Step | Action | Fingers to Show |
|------|--------|----------------|
| Step 1 | Show **NUM1** | Hold fingers still for 3 seconds |
| Step 2 | Show **OPERATOR** | 1 = ➕ &nbsp; 2 = ➖ &nbsp; 3 = ✖️ &nbsp; 4 = ➗ |
| Step 3 | Show **NUM2** | Hold fingers still for 3 seconds |
| Step 4 | **Result** displayed | Press `r` to restart |

> **Tip:** Both hands are supported! You can show numbers from 0 to 10 by combining both hands.

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `r` | Restart / New Calculation |
| `q` | Quit the application |

---

## 📁 Project Structure

```
Finger Calculator/FingerCalc
│
├── finger.py              # Main application file
├── screenshot.png         # Project screenshot
└── README.md              # Project documentation
```

---

## ⚙️ Configuration

You can tweak these settings inside the code:

```python
STABLE_THRESHOLD = 90   # ~3 seconds at 30fps (increase for slower confirmation)
min_detection_confidence = 0.7   # Hand detection sensitivity
min_tracking_confidence = 0.7    # Hand tracking sensitivity
```

---

## 🚀 Future Improvements

- [ ] Support decimal numbers
- [ ] Add voice feedback for results
- [ ] GUI-based interface
- [ ] Support for larger numbers via multi-step input
- [ ] Gesture-based undo / history

---

## 👨‍💻 Author

**Md-Kaif534 (Kaif Kohli 18)**  
GitHub: [@Md-Kaif534](https://github.com/Md-Kaif534)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

> ⭐ If you like this project, don't forget to give it a star on GitHub!
