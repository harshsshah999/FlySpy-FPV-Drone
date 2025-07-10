Absolutely! Here's the **clean, corrected, and pastable version** of your GitHub README:

---

<a name="home"></a>

# **FlySpy – AI-Enabled FPV Spy Drone**

**Open-source FPV AI Spy Drone Platform for Reconnaissance and Research**
A modular, cost-effective drone build with real-time video streaming, person detection, and BLE telemetry—designed for teaching, experimentation, and field deployment.

[Get Started →](#getting-started)

---

<a name="about"></a>

## 🔍 About the Project

This project was developed by the **FlySpy** group as part of the **Master’s Project – Intelligent Systems (SS2025)** at **Frankfurt University of Applied Sciences**.

### 🎯 Objective

Investigate and document a **low-cost, AI-capable FPV drone** system suitable for education, research, and real-world reconnaissance scenarios.

### 🧩 Project Scope

1. **Assemble** a 5-inch FPV drone using accessible components
2. **Integrate** lightweight, real-time person detection using MobileNet-SSD
3. **Display** GPS and orientation telemetry from the drone using BLE
4. **Design** a browser-based dashboard for video and sensor fusion
5. **Ensure** reproducibility, modularity, and future scalability

---

<a name="build"></a>

## 🔧 Reference Build Components

| Component                | Description                                                                 |
| ------------------------ | --------------------------------------------------------------------------- |
| Frame                    | FlyFishRC Volador II VX5 O3 FPV Freestyle T700 Frame Kit (5 inch)           |
| Flight Controller (FC)   | SpeedyBee F405 AIO 40A Bluejay (3–6S)                                       |
| Video Transmitter (VTX)  | SpeedyBee TX800 Analog FPV VTX                                              |
| Camera                   | Caddx Ratel Pro Analog FPV Camera (1500TVL)                                 |
| Receiver                 | SpeedyBee Nano 2 2.4GHz ELRS                                                |
| Remote Controller        | Radiomaster BOXER + battery                                                 |
| Charger                  | SkyRC S100 Neo (1–6S LiPo, 10A, 100W AC)                                    |
| Smoke Stopper            | TBS Smoke Stopper 2–8S                                                      |
| FPV Goggles (for stream) | Skyzone Cobra X V4 with RCA video out (for AI integration via capture card) |

---

<a name="getting-started"></a>

## 🚁 Getting Started

### 🛠️ Hardware Setup

* Assemble the frame and install motors, ESCs, and the flight controller
* Connect the ELRS receiver and verify power safety with a smoke stopper
* Power the system with a 3–6S LiPo battery
* Pair the drone with FPV goggles (e.g., Skyzone Cobra X V4 with RCA-out)

---

### ⚙️ Firmware Configuration

* Flash **Betaflight** (initially) or **ArduPilot** (for advanced GPS/autonomous features)
* Configure receiver protocol, motor direction, video output, and PID tuning
* Set failsafe behavior and assign flight modes

---

### 📺 Viewing the FPV Stream on Mac/Linux

* RCA video output from FPV goggles → RCA-to-USB capture card → MacBook
* View the stream using **OBS** or **QuickTime** via UVC input

---

<a name="ai"></a>

## 🧠 AI Integration

### 🎯 Goal

Enable **real-time person detection** on the FPV stream to simulate reconnaissance and surveillance capabilities.

### 🔧 Detection Pipeline

* **Input:** Analog video via USB capture card
* **Model:** MobileNet-SSD (via OpenCV `cv2.dnn`)
* **Processing:** Person detection and ID tracking on each frame
* **Output:** Live video with bounding boxes and ID overlays

### 🖥️ Dashboard Features

We built a browser-based dashboard using **Flask**, **OpenCV**, **HTML/CSS**, and **SocketIO** to:

* Stream video and overlay AI detection
* Display live BLE-based telemetry (GPS, orientation)
* Scan for and connect to nearby drones
* Show heading via compass and location via map
* Operate fully in-browser (tested on Chrome and Safari)

```bash
# Setup example
python3 -m venv venv
source venv/bin/activate
pip install flask flask-socketio opencv-python bleak eventlet
python app.py
```

---

Let me know if you'd like to add sections like **System Architecture**, **Screenshots**, **License**, or **Contribution Guidelines**!
