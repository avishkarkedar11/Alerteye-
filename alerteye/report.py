"""
AlertEye AI - Trip Safety Report
====================================

Builds the end-of-session summary described in spec section 9:

    Trip Safety Rating: 72/100 - Moderate Risk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .database import EventLogger


RATING_LABELS = [
    (85, "Excellent - Low Risk"),
    (70, "Good - Low-Moderate Risk"),
    (55, "Moderate Risk"),
    (35, "High Risk"),
    (0, "Severe Risk - Immediate Attention Needed"),
]


def rating_label(score: float) -> str:
    for threshold, label in RATING_LABELS:
        if score >= threshold:
            return label
    return RATING_LABELS[-1][1]


@dataclass
class TripReport:
    session_id: str
    driver_name: str
    duration_sec: float
    average_risk_score: float
    max_risk_level: str          # worst (lowest-score) alert level reached
    min_risk_score: float
    total_drowsiness_events: int
    total_yawns: int
    total_distraction_events: int
    total_warnings: int          # CAUTION-level events
    critical_events: int
    avg_driver_response_time: float | None
    overall_safety_rating: float
    rating_label: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def as_text(self) -> str:
        mins, secs = divmod(int(self.duration_sec), 60)
        lines = [
            "=" * 45,
            "        ALERTEYE AI - TRIP SAFETY REPORT",
            "=" * 45,
            f"Driver:                 {self.driver_name}",
            f"Session ID:             {self.session_id}",
            f"Duration:               {mins} min {secs} sec",
            f"Average Risk Score:     {self.average_risk_score:.1f}/100",
            f"Lowest Risk Score:      {self.min_risk_score:.1f}/100",
            f"Worst Alert Level:      {self.max_risk_level}",
            f"Drowsiness Events:      {self.total_drowsiness_events}",
            f"Yawns:                  {self.total_yawns}",
            f"Distraction Events:     {self.total_distraction_events}",
            f"Warnings (Caution+):    {self.total_warnings}",
            f"Critical Events:        {self.critical_events}",
            "Avg Driver Response:    " + (
                f"{self.avg_driver_response_time:.1f} sec"
                if self.avg_driver_response_time is not None else "N/A"
            ),
            "-" * 45,
            f"Trip Safety Rating: {self.overall_safety_rating:.0f}/100 - {self.rating_label}",
            "=" * 45,
        ]
        return "\n".join(lines)


_ALERT_SEVERITY = {"NORMAL": 0, "CAUTION": 1, "DROWSY": 2, "CRITICAL": 3}


def generate_trip_report(logger: EventLogger, session_id: str | None = None) -> TripReport:
    sid = session_id or logger.session_id
    if not sid:
        raise ValueError("No session_id available -- call start_session() first")

    info = logger.get_session_info(sid)
    events = logger.get_session_events(sid)

    driver_name = info["driver_name"] if info else "Unknown"
    start = info["start_time"] if info else None
    end = info["end_time"] if info else None

    duration_sec = 0.0
    if start and end:
        duration_sec = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()

    if not events:
        return TripReport(
            session_id=sid, driver_name=driver_name, duration_sec=duration_sec,
            average_risk_score=100.0, max_risk_level="NORMAL", min_risk_score=100.0,
            total_drowsiness_events=0, total_yawns=0, total_distraction_events=0,
            total_warnings=0, critical_events=0, avg_driver_response_time=None,
            overall_safety_rating=100.0, rating_label=rating_label(100.0),
        )

    scores = [e["risk_score"] for e in events]
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)

    worst_level = "NORMAL"
    for e in events:
        if _ALERT_SEVERITY.get(e["alert_level"], 0) > _ALERT_SEVERITY.get(worst_level, 0):
            worst_level = e["alert_level"]

    drowsiness = sum(1 for e in events if e["event_type"] == "Drowsiness")
    yawns = sum(1 for e in events if e["event_type"] == "Yawning")
    distraction = sum(1 for e in events if e["event_type"] == "Distraction")
    warnings = sum(1 for e in events if _ALERT_SEVERITY.get(e["alert_level"], 0) >= 1)
    critical = sum(1 for e in events if e["alert_level"] == "CRITICAL")

    response_times = [e["response_time"] for e in events if e["response_time"] is not None]
    avg_response = (sum(response_times) / len(response_times)) if response_times else None

    return TripReport(
        session_id=sid,
        driver_name=driver_name,
        duration_sec=duration_sec,
        average_risk_score=avg_score,
        max_risk_level=worst_level,
        min_risk_score=min_score,
        total_drowsiness_events=drowsiness,
        total_yawns=yawns,
        total_distraction_events=distraction,
        total_warnings=warnings,
        critical_events=critical,
        avg_driver_response_time=avg_response,
        overall_safety_rating=avg_score,
        rating_label=rating_label(avg_score),
    )
