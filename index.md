<a name="home"></a>
# FlySpy-FPV-Drone  

**Mission:** Document a turnkey FPV-AI drone platform—hardware, firmware, video capture, and real-time vision—so anyone can reproduce or extend it.

A reproducible, open-source FPV-AI drone platform. Below is a reference build for a 5-inch FPV drone with onboard AI capabilities.

## Reference Build Components

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
