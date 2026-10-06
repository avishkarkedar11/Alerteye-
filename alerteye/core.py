"""
AlertEye AI - Core Driver Monitoring Engine
=============================================

This module implements the AI/computer-vision core described in the
AlertEye AI spec:

    Camera -> Facial Landmarks -> EAR / MAR / Head Pose / Blink / Yawn
           -> Drowsiness & Distraction Engine -> Risk Engine (0-100)
           -> status string (NORMAL / CAUTION / DROWSY / CRITICAL)

This module is pure AI/CV -- it has no hardware or database dependency.
See `alerteye/hardware.py` for the ESP32 link and `main.py` / `dashboard.py`
for how everything is wired together.

Requirements
------------
    pip install mediapipe opencv-python numpy

Model file
----------
This uses MediaPipe's newer Tasks API (`FaceLandmarker`), which is what
current `pip install mediapipe` ships (the old `mp.solutions.face_mesh`
API has been removed from recent releases). You need the landmarker
model file once, saved as `face_landmarker.task` in the project root:

    curl -L -o face_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

from __future__ import annotations

import os
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ======================================================================
# 1. CONFIG / THRESHOLDS
# ======================================================================

class Config:
    # --- Eye Aspect Ratio ---
    EAR_THRESHOLD = 0.21          # below this = eye considered closed
    EAR_CONSEC_FRAMES_BLINK = 2   # min frames closed to count as a blink
    EYE_CLOSED_WARNING_SEC = 0.8  # closed this long -> "warning" territory
    EYE_CLOSED_DANGER_SEC = 1.6   # closed this long -> "danger" territory

    # --- Mouth Aspect Ratio (yawn) ---
    MAR_THRESHOLD = 0.6
    YAWN_CONSEC_FRAMES = 8        # ~ frames mouth must stay open to count

    # --- Blink pattern ---
    BLINK_RATE_WINDOW_SEC = 60.0
    BLINK_RATE_HIGH = 25          # blinks/min above this = fatigue signal
    BLINK_RATE_LOW = 5            # blinks/min below this = staring/fatigue

    # --- Head pose (degrees) ---
    HEAD_PITCH_DROP_WARNING = 15.0   # chin dropping toward chest
    HEAD_PITCH_DROP_DANGER = 25.0
    HEAD_YAW_DISTRACTION = 30.0      # looking away left/right

    # --- Distraction / absence ---
    DISTRACTION_WARNING_SEC = 2.0
    DISTRACTION_DANGER_SEC = 4.0
    FACE_ABSENT_DANGER_SEC = 2.0

    # --- Risk engine ---
    SCORE_MAX = 100
    SCORE_MIN = 0
    # points lost per second a bad condition persists (per signal)
    DECAY_EYE_CLOSURE = 18.0
    DECAY_YAWN = 6.0
    DECAY_HEAD_DROP = 14.0
    DECAY_DISTRACTION = 10.0
    DECAY_BLINK_ANOMALY = 4.0
    DECAY_FACE_ABSENT = 22.0
    RECOVERY_RATE = 12.0          # points regained per second when clean

    # --- Status bands (matches project spec table) ---
    BAND_ALERT = (80, 100)
    BAND_CAUTION = (60, 79)
    BAND_DROWSY = (40, 59)
    BAND_CRITICAL = (0, 39)

    # --- Response timer (section 6 of spec) ---
    RESPONSE_TIMEOUT_SEC = 6.0    # time given to recover after CRITICAL


# ======================================================================
# 2. LANDMARK INDICES (MediaPipe FaceMesh / FaceLandmarker, 478 pts)
# ======================================================================

# Six-point eye contours used for the classic Soukupova & Cech EAR formula
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth points for MAR: vertical (inner lips) + horizontal (corners)
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# Points used for solvePnP head-pose estimation
POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 263,
    "right_eye_corner": 33,
    "left_mouth_corner": 291,
    "right_mouth_corner": 61,
}

# Generic 3D face model (mm), roughly matching an average adult face,
# in the same order as POSE_LANDMARKS above.
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -63.6, -12.5),      # chin
    (-43.3, 32.7, -26.0),     # left eye corner
    (43.3, 32.7, -26.0),      # right eye corner
    (-28.9, -28.9, -24.1),    # left mouth corner
    (28.9, -28.9, -24.1),     # right mouth corner
], dtype=np.float64)


# ======================================================================
# 3. GEOMETRY HELPERS
# ======================================================================

def _dist(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def eye_aspect_ratio(pts: list[tuple[float, float]]) -> float:
    """Standard 6-point EAR. pts must be ordered like LEFT_EYE/RIGHT_EYE."""
    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = _dist(p2, p6)
    vertical_2 = _dist(p3, p5)
    horizontal = _dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(top, bottom, left, right) -> float:
    vertical = _dist(top, bottom)
    horizontal = _dist(left, right)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def estimate_head_pose(landmarks_px, frame_shape) -> tuple[float, float, float]:
    """Returns (pitch, yaw, roll) in degrees using solvePnP.

    Positive pitch  -> head tilting down (chin drop)
    Positive yaw    -> head turning to the driver's right
    """
    h, w = frame_shape[:2]
    image_points = np.array([
        landmarks_px[POSE_LANDMARKS["nose_tip"]],
        landmarks_px[POSE_LANDMARKS["chin"]],
        landmarks_px[POSE_LANDMARKS["left_eye_corner"]],
        landmarks_px[POSE_LANDMARKS["right_eye_corner"]],
        landmarks_px[POSE_LANDMARKS["left_mouth_corner"]],
        landmarks_px[POSE_LANDMARKS["right_mouth_corner"]],
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    proj_matrix = np.hstack((rotation_mat, np.zeros((3, 1))))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = [float(a[0]) for a in euler_angles]
    return pitch, yaw, roll


# ======================================================================
# 4. TEMPORAL TRACKERS
# ======================================================================

@dataclass
class BlinkTracker:
    """Counts blinks and computes a rolling blink rate."""
    consec_closed: int = 0
    total_blinks: int = 0
    blink_timestamps: deque = field(default_factory=deque)

    def update(self, eyes_closed: bool, now: float) -> None:
        if eyes_closed:
            self.consec_closed += 1
        else:
            if self.consec_closed >= Config.EAR_CONSEC_FRAMES_BLINK:
                self.total_blinks += 1
                self.blink_timestamps.append(now)
            self.consec_closed = 0
        # drop timestamps outside the rolling window
        while (self.blink_timestamps and
               now - self.blink_timestamps[0] > Config.BLINK_RATE_WINDOW_SEC):
            self.blink_timestamps.popleft()

    @property
    def blink_rate_per_min(self) -> float:
        if not self.blink_timestamps:
            return 0.0
        span = max(1.0, min(Config.BLINK_RATE_WINDOW_SEC, time.time() - self.blink_timestamps[0]))
        return len(self.blink_timestamps) * (60.0 / span)


@dataclass
class DurationTracker:
    """Generic 'how long has condition X been true' stopwatch."""
    active_since: Optional[float] = None

    def update(self, condition: bool, now: float) -> float:
        """Returns seconds the condition has been continuously true (0 if false)."""
        if condition:
            if self.active_since is None:
                self.active_since = now
            return now - self.active_since
        self.active_since = None
        return 0.0


@dataclass
class YawnTracker:
    consec_open: int = 0
    total_yawns: int = 0
    _counted_this_yawn: bool = False

    def update(self, mouth_open: bool) -> bool:
        """Returns True the frame a new yawn is confirmed."""
        if mouth_open:
            self.consec_open += 1
            if (self.consec_open >= Config.YAWN_CONSEC_FRAMES
                    and not self._counted_this_yawn):
                self.total_yawns += 1
                self._counted_this_yawn = True
                return True
        else:
            self.consec_open = 0
            self._counted_this_yawn = False
        return False


# ======================================================================
# 5. FRAME-LEVEL SIGNAL BUNDLE
# ======================================================================

@dataclass
class FrameSignals:
    face_found: bool = False
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_avg: float = 0.0
    eyes_closed: bool = False
    mar: float = 0.0
    mouth_open: bool = False
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    distracted: bool = False


# ======================================================================
# 6. RISK ENGINE
# ======================================================================

@dataclass
class RiskEngine:
    """
    Converts continuous behavioral signals into a 0-100 driver risk score.

    Design: score decays while a bad condition persists (rate scales with
    how long it has persisted, so a 3-second eye closure hurts a lot more
    than a 0.3-second one) and recovers steadily while the driver looks
    clean. This avoids single-frame false alarms (spec section 2) while
    still reacting fast to sustained danger signals.
    """
    score: float = 100.0
    last_update: Optional[float] = None
    critical_since: Optional[float] = None
    last_status: str = "NORMAL"

    def update(
        self,
        now: float,
        eye_closed_duration: float,
        yawn_active: bool,
        head_drop_deg: float,
        distraction_duration: float,
        blink_rate: float,
        face_absent_duration: float,
    ) -> dict:
        dt = 0.0 if self.last_update is None else max(0.0, now - self.last_update)
        self.last_update = now

        penalty = 0.0

        # Eye closure: penalty scales with how long eyes have been shut
        if eye_closed_duration > 0:
            severity = min(2.5, eye_closed_duration / Config.EYE_CLOSED_WARNING_SEC)
            penalty += Config.DECAY_EYE_CLOSURE * severity * dt

        # Yawning
        if yawn_active:
            penalty += Config.DECAY_YAWN * dt

        # Head drop (falling asleep, chin dropping)
        if head_drop_deg > Config.HEAD_PITCH_DROP_WARNING:
            severity = min(2.0, head_drop_deg / Config.HEAD_PITCH_DROP_WARNING)
            penalty += Config.DECAY_HEAD_DROP * severity * dt

        # Distraction (looking away / phone / etc.)
        if distraction_duration > 0:
            severity = min(2.0, distraction_duration / Config.DISTRACTION_WARNING_SEC)
            penalty += Config.DECAY_DISTRACTION * severity * dt

        # Abnormal blink pattern (too fast or too slow/staring)
        if blink_rate > Config.BLINK_RATE_HIGH or (0 < blink_rate < Config.BLINK_RATE_LOW):
            penalty += Config.DECAY_BLINK_ANOMALY * dt

        # Driver/face not detected at all -> treat as high risk
        if face_absent_duration > 0:
            severity = min(2.0, face_absent_duration / Config.FACE_ABSENT_DANGER_SEC)
            penalty += Config.DECAY_FACE_ABSENT * severity * dt

        if penalty > 0:
            self.score -= penalty
        else:
            self.score += Config.RECOVERY_RATE * dt

        self.score = max(Config.SCORE_MIN, min(Config.SCORE_MAX, self.score))
        status = self._status_for_score(self.score)

        # Response-timer bookkeeping (spec section 6: escalate if driver
        # doesn't recover from CRITICAL within RESPONSE_TIMEOUT_SEC)
        escalated = False
        if status == "CRITICAL":
            if self.critical_since is None:
                self.critical_since = now
            elif now - self.critical_since > Config.RESPONSE_TIMEOUT_SEC:
                escalated = True
        else:
            self.critical_since = None

        self.last_status = status
        return {
            "score": round(self.score, 1),
            "status": status,
            "escalated": escalated,
        }

    @staticmethod
    def _status_for_score(score: float) -> str:
        if score >= Config.BAND_ALERT[0]:
            return "NORMAL"
        if score >= Config.BAND_CAUTION[0]:
            return "CAUTION"
        if score >= Config.BAND_DROWSY[0]:
            return "DROWSY"
        return "CRITICAL"


# ======================================================================
# 7. MAIN MONITORING LOOP
# ======================================================================

class DriverMonitor:
    def __init__(self, model_path: str = "face_landmarker.task"):
        if not os.path.exists(model_path):
            if model_path == "face_landmarker.task" or not os.path.isabs(model_path):
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                print(f"Model file '{model_path}' not found. Downloading MediaPipe face_landmarker model...")
                urllib.request.urlretrieve(url, model_path)
                print(f"Model downloaded successfully to '{model_path}'.")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        self.blink_tracker = BlinkTracker()
        self.yawn_tracker = YawnTracker()
        self.eye_closure_timer = DurationTracker()
        self.distraction_timer = DurationTracker()
        self.face_absent_timer = DurationTracker()
        self.risk_engine = RiskEngine()

        self._frame_index = 0
        self._start_time = time.time()

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[FrameSignals, dict]:
        now = time.time()
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((now - self._start_time) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        self._frame_index += 1

        sig = FrameSignals()

        if not result.face_landmarks:
            sig.face_found = False
            face_absent_dur = self.face_absent_timer.update(True, now)
            self.eye_closure_timer.update(False, now)
            self.distraction_timer.update(False, now)
            risk = self.risk_engine.update(
                now=now, eye_closed_duration=0.0, yawn_active=False,
                head_drop_deg=0.0, distraction_duration=0.0,
                blink_rate=self.blink_tracker.blink_rate_per_min,
                face_absent_duration=face_absent_dur,
            )
            return sig, risk

        sig.face_found = True
        self.face_absent_timer.update(False, now)

        landmarks = result.face_landmarks[0]
        pts_px = [(lm.x * w, lm.y * h) for lm in landmarks]

        left_eye_pts = [pts_px[i] for i in LEFT_EYE]
        right_eye_pts = [pts_px[i] for i in RIGHT_EYE]
        sig.ear_left = eye_aspect_ratio(left_eye_pts)
        sig.ear_right = eye_aspect_ratio(right_eye_pts)
        sig.ear_avg = (sig.ear_left + sig.ear_right) / 2.0
        sig.eyes_closed = sig.ear_avg < Config.EAR_THRESHOLD

        sig.mar = mouth_aspect_ratio(
            pts_px[MOUTH_TOP], pts_px[MOUTH_BOTTOM],
            pts_px[MOUTH_LEFT], pts_px[MOUTH_RIGHT],
        )
        sig.mouth_open = sig.mar > Config.MAR_THRESHOLD

        sig.pitch, sig.yaw, sig.roll = estimate_head_pose(pts_px, frame_bgr.shape)
        sig.distracted = abs(sig.yaw) > Config.HEAD_YAW_DISTRACTION

        # update trackers
        self.blink_tracker.update(sig.eyes_closed, now)
        yawn_confirmed = self.yawn_tracker.update(sig.mouth_open)
        eye_closed_dur = self.eye_closure_timer.update(sig.eyes_closed, now)
        distraction_dur = self.distraction_timer.update(sig.distracted, now)

        head_drop = max(0.0, sig.pitch)  # positive pitch ~ chin dropping

        risk = self.risk_engine.update(
            now=now,
            eye_closed_duration=eye_closed_dur,
            yawn_active=sig.mouth_open,
            head_drop_deg=head_drop,
            distraction_duration=distraction_dur,
            blink_rate=self.blink_tracker.blink_rate_per_min,
            face_absent_duration=0.0,
        )

        if yawn_confirmed:
            risk["yawn_event"] = True

        return sig, risk

    def close(self) -> None:
        self.landmarker.close()


def draw_overlay(frame, sig: FrameSignals, risk: dict, blink_count: int, yawn_count: int):
    color_map = {
        "NORMAL": (0, 200, 0),
        "CAUTION": (0, 220, 220),
        "DROWSY": (0, 140, 255),
        "CRITICAL": (0, 0, 255),
    }
    color = color_map.get(risk["status"], (255, 255, 255))

    lines = [
        f"Status: {risk['status']}   Score: {risk['score']}",
        f"Eyes: {'CLOSED' if sig.eyes_closed else 'OPEN'}  EAR: {sig.ear_avg:.2f}",
        f"Mouth: {'OPEN' if sig.mouth_open else 'CLOSED'}  MAR: {sig.mar:.2f}",
        f"Head pitch/yaw: {sig.pitch:.1f} / {sig.yaw:.1f} deg",
        f"Blinks: {blink_count}   Yawns: {yawn_count}",
        f"Face detected: {sig.face_found}",
    ]
    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 25
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 6)
    return frame

