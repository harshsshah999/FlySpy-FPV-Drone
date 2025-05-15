# TEST TEXT

---
layout: page
title: FlySpy-FPV-Drone
permalink: /
---

<a name="home"></a>
# FlySpy-FPV-Drone  

**Mission:** Document a turnkey FPV-AI drone platform—hardware, firmware, video capture, and real-time vision—so anyone can reproduce or extend it.

---

<a name="getting-started"></a>
## Getting Started

1. **Hardware wiring**  
   - Frame, motors, FC/ESC, smoke-stopper, LiPo, ELRS receiver  
2. **Flash firmware**  
   - Betaflight on F405 AIO, load default config  
3. **View FPV on Mac**  
   - Skyzone Cobra X V4 → UVC capture → QuickTime/OBS  
4. **Run Python demo**  
   ```bash
   python3 -m venv v; source v/bin/activate
   pip install ultralytics opencv-python
   python yolov5_realtime.py
