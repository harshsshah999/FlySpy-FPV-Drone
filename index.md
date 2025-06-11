<a name="home"></a>
# FlySpy – AI-Enabled FPV Spy Drone

**Open-source FPV AI Spy Drone Platform for Reconnaissance and Research**  
A modular, cost-effective drone build with real-time video, AI detection, and extensible architecture—designed for teaching, experimentation, and deployment.

[Get Started →](#getting-started)

---

<a name="about"></a>
## About the Project

This project was developed by the **FlySpy** group as part of the **Master’s Project – Intelligent Systems (SS2025)** at **Frankfurt University of Applied Sciences**.

### 🎯 Objective  
Investigate and document **cost-effective, AI-enabled drones** suitable for teaching, research, and real-world reconnaissance.

### 🧩 Project Scope  
1. **Build** a low-cost 5-inch FPV drone  
2. **Integrate** a real-time AI application (e.g., human detection)  
3. **Explore** autopilot capabilities and delivery mechanisms  
4. **Document** the process with clear reproducibility  
5. **Design** for modularity and future scaling with ease

---

<a name="build"></a>
## Reference Build Components

Here’s the final build list for our FPV drone with onboard AI capabilities:

| Component                  | Description                                                                                   |
|----------------------------|-----------------------------------------------------------------------------------------------|
| Frame                      | FlyFishRC Volador II VX5 O3 FPV Freestyle T700 Frame Kit 5 inch                              |
| Flight Controller (FC)     | SpeedyBee F405 AIO 40A Bluejay 3-6S FPV Flight Controller                                     |
| Video Transmitter (VTX)    | SpeedyBee TX800 FPV VTX                                                                      |
| Camera                     | Caddx Ratel Pro Analog FPV Kamera 1500TVL Schwarz                                            |
| Receiver                   | SpeedyBee Nano 2 4GHz ELRS receiver                                                           |
| Charger                    | SkyRC S100 Neo Ladegerät Charger LiPo 1-6s 10A 100W AC                                       |
| Smoke Stopper              | TBS smoke stopper 2-8S                                                                       |
| Remote Controller          | Radiomaster BOXER remote control + battery for the remote control                           |

---

<a name="getting-started"></a>
## Getting Started

### 🛠️ Hardware Setup
- Assemble frame, mount motors, ESCs, and FC
- Connect ELRS receiver and smoke stopper
- Power with 3–6S LiPo battery
- Use goggles with AV-out for FPV (e.g., Skyzone Cobra X V4)

### ⚙️ Firmware Configuration
- Flash Betaflight or ArduPilot onto F405 AIO
- Configure receiver, video settings, and motor direction
- Tune PID, failsafe, and flight modes as needed

---

### 📺 Viewing the FPV Stream on Mac
- RCA video output from goggles → RCA to USB capture card → MacBook
- View stream using QuickTime or OBS with UVC input

---

<a name="ai"></a>
## AI Integration

### 🎯 Goal
Enable **human detection in live video** to simulate surveillance/reconnaissance capability.

### 🔧 Pipeline
- **Input:** FPV feed via USB capture card
- **Model:** YOLOv5 with Ultralytics (Python)
- **Output:** Real-time bounding boxes drawn on video stream

### 🖥️ Dashboard
We built a simple Python + HTML/CSS dashboard to:
- Show live stream in browser
- Highlight detected humans in real time

```bash
python3 -m venv v; source v/bin/activate
pip install ultralytics opencv-python
python yolov5_realtime.py
