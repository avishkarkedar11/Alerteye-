"""
AlertEye AI - Main Application
==================================

Full pipeline (spec section 14):

    Camera -> Python AI -> Facial Landmarks -> Drowsiness & Distraction
    Engine -> Risk Engine -> ESP32 Controller -> Buzzer/LEDs/Vibration
                                               -> SQLite event log
                                               -> Trip report on exit

Run
---
    # No hardware attached yet -- simulate the ESP32 link, just print commands
    python main.py --driver "Alex" --no-hardware

    # With ESP32 plugged in over USB
    python main.py --driver "Alex" --port /dev/ttyUSB0
    python main.py --driver "Alex" --port COM5       # Windows

    # List available serial ports
    python main.py --list-ports
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from alerteye.core import DriverMonitor, draw_overlay, Config
from alerteye.hardware import ESP32Link, list_available_ports
from alerteye.database import EventLogger, EventRecord
from alerteye.report import generate_trip_report


def classify_event_type(sig) -> str | None:
    """Maps a frame's signals to the event_type string used in the log."""
    if not sig.face_found:
        return "FaceAbsent"
    if sig.eyes_closed:
        return "Drowsiness"
    if sig.mouth_open:
        return "Yawning"
    if sig.distracted:
        return "Distraction"
    return None


def run(args: argparse.Namespace) -> None:
    monitor = DriverMonitor(model_path=args.model)
    hardware = ESP32Link(port=None if args.no_hardware else args.port,
                          simulate=args.no_hardware)
    hardware.start()

    logger = EventLogger(db_path=args.db)
    session_id = logger.start_session(driver_name=args.driver)
    print(f"[Session started] id={session_id} driver={args.driver}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {args.camera}", file=sys.stderr)
        sys.exit(1)

    last_sent_status = None
    last_logged_event = None
    last_event_start = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            sig, risk = monitor.process_frame(frame)
            now = time.time()

            # -- hardware: only push a new command when status changes --
            if risk["status"] != last_sent_status:
                hardware.send_command(risk["status"])
                last_sent_status = risk["status"]
            if risk.get("escalated"):
                hardware.send_command("CRITICAL_ESCALATE")

            # -- logging: log a new row whenever the event type changes --
            event_type = classify_event_type(sig)
            if event_type != last_logged_event:
                if last_logged_event is not None and last_event_start is not None:
                    duration = now - last_event_start
                    if duration >= 0.3:  # ignore flicker-short events
                        logger.log_event(EventRecord(
                            event_type=last_logged_event,
                            risk_score=risk["score"],
                            alert_level=risk["status"],
                            duration=round(duration, 2),
                            driver_response="Recovered" if event_type is None else None,
                        ))
                last_logged_event = event_type
                last_event_start = now if event_type else None

            if not args.no_display:
                draw_overlay(
                    frame, sig, risk,
                    monitor.blink_tracker.total_blinks,
                    monitor.yawn_tracker.total_yawns,
                )
                hw = hardware.status()
                cv2.putText(
                    frame, f"ESP32: {'CONNECTED' if hw.connected else 'DISCONNECTED'}",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if hw.connected else (0, 0, 255), 2,
                )
                cv2.imshow("AlertEye AI - Driver Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print(f"{risk['status']:9s} score={risk['score']:5.1f}  "
                      f"eyes={'CLOSED' if sig.eyes_closed else 'OPEN ':6s}  "
                      f"blinks={monitor.blink_tracker.total_blinks}  "
                      f"yawns={monitor.yawn_tracker.total_yawns}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        monitor.close()
        hardware.stop()
        logger.end_session()

        report = generate_trip_report(logger, session_id)
        print("\n" + report.as_text())
        logger.close()


def main():
    parser = argparse.ArgumentParser(description="AlertEye AI - full pipeline")
    parser.add_argument("--driver", type=str, default="Unknown", help="Driver name for the session log")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=str, default="face_landmarker.task")
    parser.add_argument("--db", type=str, default="alerteye.db")
    parser.add_argument("--port", type=str, default=None, help="ESP32 serial port, e.g. /dev/ttyUSB0 or COM5")
    parser.add_argument("--no-hardware", action="store_true", help="Run without a physical ESP32 (simulated)")
    parser.add_argument("--no-display", action="store_true", help="Headless mode, no OpenCV window")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")
    args = parser.parse_args()

    if args.list_ports:
        ports = list_available_ports()
        print("Available serial ports:" if ports else "No serial ports found.")
        for p in ports:
            print(f"  {p}")
        return

    if not args.no_hardware and not args.port:
        print("No --port given and --no-hardware not set -- running in simulated "
              "hardware mode. Pass --port <device> once the ESP32 is connected.")
        args.no_hardware = True

    run(args)


if __name__ == "__main__":
    main()
