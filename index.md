<a name="home"></a>
# FlySpy-FPV-Drone  

**Mission:** Document a turnkey FPV-AI drone platform—hardware, firmware, video capture, and real-time vision—so anyone can reproduce or extend it.

A reproducible, open-source FPV-AI drone platform. Below is a reference build for a 3.5-inch FPV drone with onboard AI capabilities.

## Reference Build Components

| Item                        | Options                                                                                       |
|-----------------------------|-----------------------------------------------------------------------------------------------|
| Drone Frame                 | 3.5‑inch FPV frame (e.g., DarwinFPV BabyApe II, FlyFishRC Volador VX3.5, SpeedyBee Bee35 PRO) |
| Flight Controller (FC) + ESC| Lightweight FC for 3.5‑inch drones (e.g., SpeedyBee F405 AIO or similar)                      |
| FPV Video Transmitter (VTX) | Analog VTX (e.g., SpeedyBee TX800 FPV VTX designed for low‑power, light setups)               |
| FPV Camera                  | Analog FPV camera (e.g., Caddx Ratel Pro – lightweight and compatible with 3.5‑inch builds)    |
| Receiver (FPV)              | ELRS receiver (e.g., Radiomaster RP1 ELRS 2.4GHz)                                              |
| Remote Controller           | Radiomaster Pocket ELRS RC                                                                     |
| FPV Goggles                 | Analog goggles with AV‑out (e.g., Fat Shark ECHO or Skyzone CobraX V4)                        |
| Battery (LiPo)              | S4 LiPo battery (capacity chosen to balance flight time and weight for a 3.5‑inch drone)      |
| Battery Charger             | Compatible LiPo charger (typically in the €50–€60 range)                                       |
| Smoke Stopper               | TBS Smoke Stopper or Vifly ShortSafer V2                                                      |
| Onboard Computer            | Raspberry Pi Zero 2 W (ideal lightweight onboard computer for sub‑250g drones)                 |
| AI Accelerator              | Google Coral USB Accelerator (for fast, onboard AI inference)                                  |
| Ground Control Station (GCS)| QGroundControl or Mission Planner (for telemetry, flight planning, and monitoring)             |
| Flight Controller Firmware  | Betaflight (for manual FPV) or PX4/ArduPilot (if integrating autonomous functions)             |
| Documentation Tools         | GitHub Pages, Markdown editors                                                                 |
| Motor                       | iFlight XING2 1404 FPV Motor 3800KV<br>T-Motor P1604 Freestyle Sub FPV Motor 3800KV Silver    |
|                             | 150mm or smaller 3″ or smaller 1105 -1306 or smaller 3000KV and higher                        |
|                             | 180mm 4″1806, 2204 2600KV – 3000KV                                                            |
| Propellor                   | HQProp T3.5X2X3 3.5 inch 3-blade propeller gray (2CW+2CCW)                                    |

---

*For more details, see the project repository and documentation.*

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
```

<!-- Custom footer: intentionally left blank -->