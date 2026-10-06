"""
AlertEye AI - ESP32 Hardware Link
====================================

Handles serial communication with the ESP32 (spec section 5):

    Camera -> Python AI -> Risk Analysis -> ESP32 -> Hardware Alert

The Python side sends one of: NORMAL, CAUTION, DROWSY, CRITICAL,
CRITICAL_ESCALATE, and reads back single-character acknowledgements the
firmware sends (e.g. "A" when the driver presses the physical
acknowledge button -- see firmware/alerteye_esp32/alerteye_esp32.ino).

This module never lets a hardware problem crash the AI loop: every
serial operation is wrapped, connection loss is detected, and the link
auto-retries in the background.

Requirements
------------
    pip install pyserial
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

try:
    import serial
    import serial.tools.list_ports
except ImportError:  # pyserial not installed
    serial = None


VALID_COMMANDS = {"NORMAL", "CAUTION", "DROWSY", "CRITICAL", "CRITICAL_ESCALATE"}

# The firmware has its own independent timeout watchdog (spec section 12),
# but the Python side also re-sends the current status periodically so a
# silent link never looks "connected" on our end either.
HEARTBEAT_INTERVAL_SEC = 1.0
RECONNECT_INTERVAL_SEC = 3.0


@dataclass
class HardwareStatus:
    connected: bool = False
    port: Optional[str] = None
    last_command: Optional[str] = None
    last_ack: Optional[str] = None
    last_send_time: float = 0.0
    last_error: Optional[str] = None


def list_available_ports() -> list[str]:
    """Convenience helper for a setup UI / CLI --list-ports flag."""
    if serial is None:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


class ESP32Link:
    """
    Thread-safe serial link to the ESP32.

    Usage:
        link = ESP32Link(port="/dev/ttyUSB0")   # or "COM5" on Windows
        link.start()
        link.send_command("DROWSY")
        ...
        link.stop()

    If `port` is None, the link runs in simulation mode: no real serial
    port is opened, commands are just logged. This lets the AI + dashboard
    + database modules all be developed and demoed without hardware
    attached.
    """

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200,
                 timeout: float = 1.0, simulate: Optional[bool] = None):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.simulate = simulate if simulate is not None else (port is None or serial is None)

        self._ser = None
        self._lock = threading.Lock()
        self._status = HardwareStatus(port=port)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self.simulate:
            self._status.connected = True
            print(f"[ESP32Link] Simulation mode (no hardware) -- "
                  f"commands will be printed, not sent.")
            return
        self._connect()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
            self._status.connected = False

    # -- public API -------------------------------------------------------

    def send_command(self, command: str) -> bool:
        """Send a status command. Returns True if it was (or would be) sent."""
        if command not in VALID_COMMANDS:
            raise ValueError(f"Unknown command '{command}'. Must be one of {VALID_COMMANDS}")

        self._status.last_command = command
        self._status.last_send_time = time.time()

        if self.simulate:
            print(f"[ESP32 SIM] -> {command}")
            return True

        with self._lock:
            if self._ser is None or not self._ser.is_open:
                self._status.connected = False
                return False
            try:
                self._ser.write((command + "\n").encode("utf-8"))
                return True
            except Exception as exc:
                self._status.connected = False
                self._status.last_error = str(exc)
                return False

    def status(self) -> HardwareStatus:
        return self._status

    # -- internals --------------------------------------------------------

    def _connect(self) -> bool:
        if serial is None:
            self._status.last_error = "pyserial not installed"
            return False
        try:
            with self._lock:
                self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2.0)  # let the ESP32 finish its reset-on-connect
            self._status.connected = True
            self._status.last_error = None
            print(f"[ESP32Link] Connected on {self.port} @ {self.baudrate} baud")
            return True
        except Exception as exc:
            self._status.connected = False
            self._status.last_error = str(exc)
            print(f"[ESP32Link] Connection failed: {exc}")
            return False

    def _run(self) -> None:
        """Background thread: heartbeat re-sends + read acks + auto-reconnect."""
        last_heartbeat = 0.0
        last_reconnect_attempt = 0.0

        while not self._stop_event.is_set():
            now = time.time()

            if not self._status.connected:
                if now - last_reconnect_attempt > RECONNECT_INTERVAL_SEC:
                    last_reconnect_attempt = now
                    self._connect()
                time.sleep(0.2)
                continue

            # heartbeat: keep re-sending the last known status so the
            # firmware's own link-timeout watchdog stays satisfied
            if (self._status.last_command
                    and now - last_heartbeat > HEARTBEAT_INTERVAL_SEC):
                last_heartbeat = now
                self.send_command(self._status.last_command)

            # read any acknowledgement bytes from the push button
            with self._lock:
                try:
                    if self._ser and self._ser.in_waiting:
                        line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                        if line:
                            self._status.last_ack = line
                except Exception as exc:
                    self._status.connected = False
                    self._status.last_error = str(exc)

            time.sleep(0.05)
