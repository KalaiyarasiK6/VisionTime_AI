"""
activity.py
-----------
Intelligent Event Engine for VisionTime AI.

Generates a generic, environment-agnostic Event Timeline purely from
TRACKING DATA: position, zone membership, dwell time, and movement speed.

There is NO appearance-based "activity recognition" (walking / running /
standing). Zones are plain spatial grid cells (Zone A, Zone B, ...) with no
semantic meaning, so this works unmodified for airports, hospitals,
colleges, offices, malls, warehouses, government buildings, factories,
parking areas, or any other CCTV scene.

Events produced:
    - Entered Scene
    - Exited Scene
    - Moved to Zone X
    - Stayed in Zone X
    - Long Stay
    - Loitering
    - Crowd Detected
    - Fast Movement
"""

import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Zone Grid
# ---------------------------------------------------------------------------

class ZoneGrid:
    """
    Divides the video frame into a generic grid of zones (Zone A, Zone B, ...).
    Purely spatial - carries no semantic meaning, so it works for any
    CCTV environment without hardcoding shelves, counters, etc.
    """

    def __init__(self, frame_width: int, frame_height: int, rows: int = 1, cols: int = 5):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.rows = max(1, rows)
        self.cols = max(1, cols)
        self.cell_w = frame_width / self.cols
        self.cell_h = frame_height / self.rows
        self.labels = list(string.ascii_uppercase[: self.rows * self.cols])

    def zone_for_point(self, x: float, y: float) -> str:
        col = min(max(int(x // self.cell_w), 0), self.cols - 1)
        row = min(max(int(y // self.cell_h), 0), self.rows - 1)
        idx = row * self.cols + col
        return f"Zone {self.labels[idx]}"

    def boundaries(self) -> List[Tuple[int, int, int, int, str]]:
        """Return list of (x1, y1, x2, y2, label) rectangles for drawing overlays."""
        rects = []
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = int(c * self.cell_w)
                y1 = int(r * self.cell_h)
                x2 = int((c + 1) * self.cell_w)
                y2 = int((r + 1) * self.cell_h)
                idx = r * self.cols + c
                rects.append((x1, y1, x2, y2, f"Zone {self.labels[idx]}"))
        return rects


# ---------------------------------------------------------------------------
# Per-track state
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    track_id: int
    first_seen_ts: float
    last_seen_ts: float
    last_seen_frame: int
    current_zone: str
    zone_entered_ts: float
    last_centroid: Tuple[float, float]
    exited: bool = False
    long_stay_fired: bool = False
    loitering_fired: bool = False
    fast_move_cooldown_until: float = 0.0
    positions_in_zone: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class Event:
    timestamp: float          # seconds from start of video
    track_id: Optional[int]
    event_type: str
    zone: Optional[str]
    description: str

    def timestamp_str(self) -> str:
        m, s = divmod(int(self.timestamp), 60)
        return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Event Engine
# ---------------------------------------------------------------------------

class EventEngine:
    """
    Consumes per-frame tracked-person lists and produces a stream of
    intelligent, generic timeline events. Nothing here is hardcoded to a
    specific environment - only geometry, time, and speed are used.
    """

    def __init__(
        self,
        zone_grid: ZoneGrid,
        fps: float,
        stay_seconds: float = 3.0,
        long_stay_seconds: float = 10.0,
        loitering_seconds: float = 20.0,
        loitering_move_px: float = 40.0,
        fast_move_px_per_sec: float = 300.0,
        crowd_size: int = 4,
        exit_grace_seconds: float = 1.5,
    ):
        self.zone_grid = zone_grid
        self.fps = fps
        self.stay_seconds = stay_seconds
        self.long_stay_seconds = long_stay_seconds
        self.loitering_seconds = loitering_seconds
        self.loitering_move_px = loitering_move_px
        self.fast_move_px_per_sec = fast_move_px_per_sec
        self.crowd_size = crowd_size
        self.exit_grace_seconds = exit_grace_seconds

        self.tracks: Dict[int, TrackState] = {}
        self.events: List[Event] = []
        self._crowd_active: Dict[str, bool] = {}
        self._stay_fired: Dict[int, str] = {}

    # -- internal helpers -----------------------------------------------

    def _add_event(self, ts: float, track_id: Optional[int], event_type: str,
                    zone: Optional[str], description: str):
        self.events.append(Event(ts, track_id, event_type, zone, description))

    # -- main update -------------------------------------------------------

    def update(self, frame_idx: int, timestamp: float, people: List) -> None:
        seen_ids = set()
        zone_occupancy: Dict[str, List[int]] = {}

        for person in people:
            tid = person.track_id
            seen_ids.add(tid)
            zone = self.zone_grid.zone_for_point(*person.centroid)
            zone_occupancy.setdefault(zone, []).append(tid)

            if tid not in self.tracks or self.tracks[tid].exited:
                # Either a brand-new person, OR a person who was previously
                # marked "Exited Scene" but has now reappeared (tracker
                # remembered their ID across a brief absence). Either way,
                # this must start a clean new "Entered Scene" -> zone entry
                # sequence rather than silently resuming stale state - that
                # is what previously produced contradictory timelines like
                # "Exited Scene" immediately followed by "Moved to Zone D".
                state = TrackState(
                    track_id=tid,
                    first_seen_ts=timestamp,
                    last_seen_ts=timestamp,
                    last_seen_frame=frame_idx,
                    current_zone=zone,
                    zone_entered_ts=timestamp,
                    last_centroid=person.centroid,
                )
                self.tracks[tid] = state
                self._stay_fired.pop(tid, None)
                self._add_event(timestamp, tid, "entered_scene", zone,
                                 f"Person {tid} Entered Scene")
                self._add_event(timestamp, tid, "zone_change", zone,
                                 f"Person {tid} Entered {zone}")
                continue

            state = self.tracks[tid]

            dt = max(timestamp - state.last_seen_ts, 1e-6)
            dx = person.centroid[0] - state.last_centroid[0]
            dy = person.centroid[1] - state.last_centroid[1]
            dist = (dx ** 2 + dy ** 2) ** 0.5
            speed = dist / dt

            if speed >= self.fast_move_px_per_sec and timestamp >= state.fast_move_cooldown_until:
                self._add_event(timestamp, tid, "fast_movement", state.current_zone,
                                 f"Person {tid} Fast Movement in {state.current_zone}")
                state.fast_move_cooldown_until = timestamp + 3.0

            if zone != state.current_zone:
                self._add_event(timestamp, tid, "zone_change", zone,
                                 f"Person {tid} Moved to {zone}")
                state.current_zone = zone
                state.zone_entered_ts = timestamp
                state.long_stay_fired = False
                state.loitering_fired = False
                state.positions_in_zone = [person.centroid]
                self._stay_fired.pop(tid, None)
            else:
                state.positions_in_zone.append(person.centroid)
                dwell = timestamp - state.zone_entered_ts

                if dwell >= self.stay_seconds and self._stay_fired.get(tid) != zone:
                    self._add_event(timestamp, tid, "stayed", zone,
                                     f"Person {tid} Stayed in {zone}")
                    self._stay_fired[tid] = zone

                if dwell >= self.long_stay_seconds and not state.long_stay_fired:
                    self._add_event(timestamp, tid, "long_stay", zone,
                                     f"Person {tid} Long Stay in {zone} ({int(dwell)}s)")
                    state.long_stay_fired = True

                if dwell >= self.loitering_seconds and not state.loitering_fired:
                    xs = [p[0] for p in state.positions_in_zone]
                    ys = [p[1] for p in state.positions_in_zone]
                    spread = (max(xs) - min(xs)) + (max(ys) - min(ys))
                    if spread <= self.loitering_move_px * 4:
                        self._add_event(timestamp, tid, "loitering", zone,
                                         f"Person {tid} Loitering in {zone} ({int(dwell)}s)")
                        state.loitering_fired = True

            state.last_seen_ts = timestamp
            state.last_seen_frame = frame_idx
            state.last_centroid = person.centroid

        # Crowd detection (fires once per crowd episode, per zone)
        for zone, ids in zone_occupancy.items():
            is_crowd = len(ids) >= self.crowd_size
            was_crowd = self._crowd_active.get(zone, False)
            if is_crowd and not was_crowd:
                self._add_event(timestamp, None, "crowd_detected", zone,
                                 f"Crowd Detected in {zone} ({len(ids)} persons)")
            self._crowd_active[zone] = is_crowd

        # Exit detection with grace period (handles brief occlusions)
        for tid, state in self.tracks.items():
            if state.exited or tid in seen_ids:
                continue
            if timestamp - state.last_seen_ts >= self.exit_grace_seconds:
                self._add_event(state.last_seen_ts, tid, "exited_scene", state.current_zone,
                                 f"Person {tid} Exited Scene")
                state.exited = True

    def finalize(self, final_timestamp: float) -> None:
        """Call once after the video ends, to close out any still-active tracks."""
        for tid, state in self.tracks.items():
            if not state.exited:
                self._add_event(final_timestamp, tid, "exited_scene", state.current_zone,
                                 f"Person {tid} Exited Scene")
                state.exited = True

    def sorted_events(self) -> List[Event]:
        return sorted(self.events, key=lambda e: e.timestamp)
