"""
Pydantic schemas for the Crime Detection API.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import time
from pydantic import BaseModel


class IncidentAlert(BaseModel):
    """Serializable incident alert for the dashboard event feed."""
    timestamp: float
    timestamp_str: str
    person_track_id: int
    weapon_class: str
    threat_score: float
    threat_level: int
    threat_label: str
    action_tags: List[str]
    person_center: Tuple[int, int]
    weapon_center: Optional[Tuple[int, int]] = None
    target_track_id: Optional[int] = None

    @classmethod
    def from_threat_event(cls, event) -> "IncidentAlert":
        import datetime
        return cls(
            timestamp=event.timestamp,
            timestamp_str=datetime.datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S"),
            person_track_id=event.person_track_id,
            weapon_class=event.weapon_class or "unknown",
            threat_score=round(event.threat_score, 1),
            threat_level=event.threat_level,
            threat_label=event.threat_label,
            action_tags=event.action_tags,
            person_center=event.person_center,
            weapon_center=event.weapon_center,
            target_track_id=event.target_track_id,
        )


class StreamStatus(BaseModel):
    """Current stream status message."""
    running: bool
    source: str
    fps: float
    frame_count: int
    incident_count: int
