# AlertEye AI
### Intelligent Driver Drowsiness, Distraction & Safety Monitoring System (AI + IoT)

Real-time driver monitoring that combines eye closure, blink pattern,
yawning, head pose, and distraction into a single 0-100 **Driver Risk
Score**, drives an ESP32-controlled buzzer/LED/vibration alert system,
logs every event to SQLite, and shows it all on a live Streamlit
dashboard.

```
Camera -> Python AI (EAR/MAR/Head Pose) -> Risk Engine (0-100)
       -> ESP32 (Buzzer/LEDs/Vibration/Display)
       -> SQLite event log -> Trip Safety Report
```

## Project layout

```
alerteye_project/
├── alerteye/
│   ├── core.py        # facial landmarks, EAR/MAR, drowsiness signals, risk engine
│   ├── hardware.py     # serial link to the ESP32 (auto-reconnect, heartbeat)
│   ├── database.py     # SQLite session + event logging
│   └── report.py       # end-of-trip safety report
├── firmware/
│   └── alerteye_esp32/alerteye_esp32.ino   # ESP32 Arduino sketch
├── main.py              # CLI: camera window + full pipeline
├── dashboard.py          # Streamlit live dashboard
└── requirements.txt
```

## 1. Software setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Download the MediaPipe face landmark model once (the newer MediaPipe
Tasks API needs this file locally — it's not bundled with pip install):

```bash
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Put `face_landmarker.task` in the project root (next to `main.py`).

## 2. Run without hardware (webcam + on-screen window)

```bash
python main.py --driver "Alex" --no-hardware
```

Press `q` in the video window to stop; a trip report prints to the
console.

## 3. Run the dashboard

```bash
streamlit run dashboard.py
```

Fill in the sidebar (driver name, camera index) and click **Start**.
Leave "Use real ESP32" unchecked to run fully in software.

## 4. Hardware setup (ESP32)

1. Open `firmware/alerteye_esp32/alerteye_esp32.ino` in the Arduino IDE
   (install the ESP32 board package via Boards Manager if you haven't).
2. Wire it up:

   | Component        | ESP32 Pin |
   |-------------------|-----------|
   | Green LED          | GPIO 25   |
   | Yellow LED         | GPIO 26   |
   | Red LED            | GPIO 27   |
   | Buzzer             | GPIO 14   |
   | Vibration motor*   | GPIO 12   |
   | Push button         | GPIO 13 (to GND) |
   | OLED (optional)     | SDA 21 / SCL 22 |

   \* Drive the vibration motor through a transistor/MOSFET, not
   directly off the GPIO pin.

3. Flash the sketch, note the serial port it enumerates as
   (`/dev/ttyUSB0`, `/dev/cu.usbserial-*`, or `COM5` on Windows —
   run `python main.py --list-ports` to check).
4. Run with hardware attached:

   ```bash
   python main.py --driver "Alex" --port /dev/ttyUSB0
   ```

The ESP32 firmware has its **own independent watchdog**: if it stops
receiving commands from the PC for 5 seconds, it fails safe into a
"SYSTEM CONNECTION LOST" state on its own (flashing LEDs + periodic
buzzer chirp) rather than silently doing nothing.

## 5. How the pieces fit together

- **`alerteye/core.py`** — pure computer vision. Turns camera frames
  into EAR, MAR, head pose, blink/yawn counts, and a continuously
  decaying/recovering 0–100 risk score with NORMAL/CAUTION/DROWSY/
  CRITICAL bands. No hardware or database dependency, so it's easy to
  unit-test on its own.
- **`alerteye/hardware.py`** — serial link only. Runs in simulation
  mode (prints commands) when no `--port` is given, so the rest of the
  app works before you've touched a breadboard.
- **`alerteye/database.py`** — SQLite session/event log.
- **`alerteye/report.py`** — reads the log back and produces the trip
  summary.
- **`main.py`** / **`dashboard.py`** — two different front ends
  (OpenCV window vs. Streamlit) that both just import and wire together
  the modules above.

## Notes on tuning

Thresholds (EAR, MAR, timing windows, risk decay/recovery rates) all
live in `Config` at the top of `alerteye/core.py`. Calibrate `EAR_THRESHOLD`
for your camera/lighting by printing `sig.ear_avg` for a minute with
eyes open vs. closed before relying on the alerts.
