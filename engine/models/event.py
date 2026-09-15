"""The Event model: one entry in the scheduler's real-time event log."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from engine.models.enums import EventType


@dataclass
class Event:
    """A single, timestamped record of something the engine did or saw.

    Events are how the frontend's event log and later phases'
    decision history are built. This model only describes the shape
    of an event - nothing here generates, filters, or reacts to
    events; that is event-processing logic for a later phase.
    """

    event_id: str
    event_type: EventType
    message: str

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Related entities are all optional and independent - an event
    # might touch a GPU and a user but no job, or just the system.
    gpu_id: Optional[str] = None
    user_id: Optional[str] = None
    job_id: Optional[str] = None

    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
