"""
AlertEye AI - Driver Safety Dashboard
=========================================

    streamlit run dashboard.py

A live camera feed + real-time metrics (spec section 8), plus a trip
report at the end of the session (spec section 9). Runs the ESP32 link
in simulation mode by default so the dashboard works without hardware --
tick "Use real ESP32" in the sidebar once it's plugged in.
"""

from __future__ import annotations

import os
import time

import cv2
import streamlit as st

from alerteye.core import DriverMonitor, Config
from alerteye.hardware import ESP32Link, list_available_ports
from alerteye.database import EventLogger, EventRecord
from alerteye.report import generate_trip_report

st.set_page_config(page_title="AlertEye AI - Driver Monitor", layout="wide")

STATUS_ICON = {"NORMAL": "🟢", "CAUTION": "🟡", "DROWSY": "🟠", "CRITICAL": "🔴"}
STATUS_LABEL = {"NORMAL": "ALERT", "CAUTION": "CAUTION", "DROWSY": "DROWSY", "CRITICAL": "CRITICAL"}


def init_state():
    defaults = dict(
        running=False, monitor=None, hardware=None, logger=None, session_id=None,
        blink_count=0, yawn_count=0, last_status=None,
        last_event_type=None, last_event_start=None,
        warning_count=0, critical_count=0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def classify_event_type(sig) -> str | None:
    if not sig.face_found:
        return "FaceAbsent"
    if sig.eyes_closed:
        return "Drowsiness"
    if sig.mouth_open:
        return "Yawning"
    if sig.distracted:
        return "Distraction"
    return None


def start_session(driver_name, camera_index, model_path, use_hardware, port, db_path):
    st.session_state.driver_name = driver_name
    st.session_state.camera_index = int(camera_index)
    st.session_state.model_path = model_path
    st.session_state.use_hardware = use_hardware
    st.session_state.port = port
    st.session_state.db_path = db_path
    st.session_state.running = True
    st.session_state.show_report = False
    st.session_state.warning_count = 0
    st.session_state.critical_count = 0
    st.session_state.last_status = None
    st.session_state.last_event_type = None
    st.session_state.last_event_start = None
    st.session_state.cap = None
    st.session_state.monitor = None
    st.session_state.hardware = None
    st.session_state.logger = None


def stop_session():
    st.session_state.running = False
    if st.session_state.get("cap"):
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None
    if st.session_state.get("monitor"):
        try:
            st.session_state.monitor.close()
        except Exception:
            pass
        st.session_state.monitor = None
    if st.session_state.get("hardware"):
        try:
            st.session_state.hardware.stop()
        except Exception:
            pass
        st.session_state.hardware = None
    if st.session_state.get("logger"):
        try:
            st.session_state.logger.end_session()
        except Exception:
            pass
        st.session_state.logger = None


def main():
    init_state()

    st.title("🚗 AlertEye AI - Driver Monitor")

    with st.sidebar:
        st.header("Session Setup")
        driver_name = st.text_input("Driver name", value="Driver 1", disabled=st.session_state.running)
        camera_index = st.number_input("Camera index", min_value=0, value=0, disabled=st.session_state.running)
        model_path = st.text_input("Face landmarker model path", value="face_landmarker.task",
                                    disabled=st.session_state.running)
        db_path = st.text_input("Database file", value="alerteye.db", disabled=st.session_state.running)

        st.divider()
        use_hardware = st.checkbox("Use real ESP32", value=False, disabled=st.session_state.running)
        port = None
        if use_hardware:
            ports = list_available_ports()
            port = st.selectbox("Serial port", options=ports or ["No ports found"],
                                 disabled=st.session_state.running)

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶ Start", disabled=st.session_state.running, use_container_width=True):
                start_session(driver_name, camera_index, model_path, use_hardware, port, db_path)
                st.rerun()
        with col_b:
            if st.button("⏹ Stop", disabled=not st.session_state.running, use_container_width=True):
                stop_session()
                st.rerun()

        if not st.session_state.running and st.session_state.get("session_id"):
            st.divider()
            if st.button("📄 Generate Trip Report", use_container_width=True):
                st.session_state.show_report = True

    if not st.session_state.running:
        st.info("Set your session options in the sidebar, then click **Start** to begin monitoring.")
        if st.session_state.get("show_report") and st.session_state.get("session_id"):
            logger = EventLogger(db_path=db_path)
            report = generate_trip_report(logger, st.session_state.session_id)
            st.subheader("Trip Safety Report")
            st.code(report.as_text())
            logger.close()
        return

    # -- Initialize Hardware, Model & Camera inside main screen flow --
    if st.session_state.cap is None:
        with st.spinner("📷 Opening camera & initializing AI model..."):
            camera_idx = st.session_state.camera_index
            cap = cv2.VideoCapture(camera_idx)
            if not cap.isOpened() and os.name == "nt":
                cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)

            if not cap.isOpened():
                st.error(f"❌ Cannot access camera at index {camera_idx}. "
                         f"If your computer has only 1 webcam (built-in), set **Camera index** to `0` in the sidebar. "
                         f"Also ensure no other program is currently using the camera.")
                stop_session()
                return

            try:
                monitor = DriverMonitor(model_path=st.session_state.model_path)
            except Exception as e:
                st.error(f"❌ Failed to load face landmarker model: {e}")
                cap.release()
                stop_session()
                return

            hardware = ESP32Link(port=st.session_state.port if st.session_state.use_hardware else None,
                                 simulate=not st.session_state.use_hardware)
            hardware.start()
            logger = EventLogger(db_path=st.session_state.db_path)
            session_id = logger.start_session(st.session_state.driver_name)

            st.session_state.cap = cap
            st.session_state.monitor = monitor
            st.session_state.hardware = hardware
            st.session_state.logger = logger
            st.session_state.session_id = session_id

    # ---------------- live monitoring layout ----------------
    col_feed, col_metrics = st.columns([2, 1])
    frame_placeholder = col_feed.empty()

    status_ph = col_metrics.empty()
    score_ph = col_metrics.empty()
    metrics_ph = col_metrics.empty()
    hardware_ph = col_metrics.empty()

    cap = st.session_state.cap
    monitor: DriverMonitor = st.session_state.monitor
    hardware: ESP32Link = st.session_state.hardware
    logger: EventLogger = st.session_state.logger

    while st.session_state.running:
        ok, frame = cap.read()
        if not ok:
            retry_ok = False
            for _ in range(3):
                time.sleep(0.05)
                ok, frame = cap.read()
                if ok:
                    retry_ok = True
                    break
            if not retry_ok:
                st.error("Camera feed disconnected or read failed. Ensure the camera is connected and not locked by another application.")
                stop_session()
                break
        frame = cv2.flip(frame, 1)

        sig, risk = monitor.process_frame(frame)
        now = time.time()

        # -- hardware --
        if risk["status"] != st.session_state.last_status:
            hardware.send_command(risk["status"])
            st.session_state.last_status = risk["status"]
            if risk["status"] == "CAUTION":
                st.session_state.warning_count += 1
            elif risk["status"] == "CRITICAL":
                st.session_state.critical_count += 1
        if risk.get("escalated"):
            hardware.send_command("CRITICAL_ESCALATE")

        # -- logging --
        event_type = classify_event_type(sig)
        if event_type != st.session_state.last_event_type:
            prev = st.session_state.last_event_type
            prev_start = st.session_state.last_event_start
            if prev is not None and prev_start is not None:
                duration = now - prev_start
                if duration >= 0.3:
                    logger.log_event(EventRecord(
                        event_type=prev, risk_score=risk["score"],
                        alert_level=risk["status"], duration=round(duration, 2),
                        driver_response="Recovered" if event_type is None else None,
                    ))
            st.session_state.last_event_type = event_type
            st.session_state.last_event_start = now if event_type else None

        # -- render feed --
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # -- render metrics (spec section 8 mockup) --
        icon = STATUS_ICON.get(risk["status"], "⚪")
        status_ph.markdown(f"### Driver Status: {icon} {STATUS_LABEL.get(risk['status'], risk['status'])}")
        score_ph.metric("Risk Score", f"{risk['score']:.0f} / 100")

        hw = hardware.status()
        metrics_ph.markdown(f"""
**Eyes:** {'CLOSED' if sig.eyes_closed else 'OPEN'}
**Blink Count:** {monitor.blink_tracker.total_blinks}
**Yawns:** {monitor.yawn_tracker.total_yawns}
**Head Direction:** {'LEFT' if sig.yaw < -15 else 'RIGHT' if sig.yaw > 15 else 'FORWARD'}
**Distraction:** {'YES' if sig.distracted else 'No'}
**Warnings:** {st.session_state.warning_count}
**Critical Events:** {st.session_state.critical_count}
        """)

        hardware_ph.markdown(f"""
---
**Buzzer:** {'ON' if risk['status'] in ('DROWSY', 'CRITICAL') else 'OFF'}
**Vibration:** {'ON' if risk['status'] == 'CRITICAL' else 'OFF'}
**ESP32:** {'CONNECTED' if hw.connected else 'DISCONNECTED'}
        """)

        time.sleep(0.03)


if __name__ == "__main__":
    main()
