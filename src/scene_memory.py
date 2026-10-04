"""Short-term scene memory: decides when a scene change is worth speaking.

VisionGuide should give useful guidance, not a live detection transcript. Each
new scene is compared with what was seen and said before:

- an obstacle that appears in the path (and stays for a few frames) is
  announced once, with which way to go around it;
- an announced obstacle that moves into a closer proximity bucket is
  announced again ("getting closer");
- the path becoming blocked on every side, or a likely drop in the ground,
  is announced once ("Stop");
- something big closing in much faster than walking pace (a bike, a car) is
  called out with the side it is coming from;
- after a stop, the path opening up again is mentioned once;
- while the user is being guided to a target ("guide me to the chair"), its
  direction is repeated every few seconds until they reach it;
- unchanged, receding or off-path objects stay silent;
- an object that disappears for a while is forgotten, so it is announced
  again if it comes back.

Modes: ``quiet`` (urgent warnings and target guidance only), ``normal``,
``verbose`` (also objects passing close on either side). ``paused`` mutes
everything while still tracking.

Urgent announcements (very close obstacle, stop, drop-off, fast approach) may
cut the agent's own speech short. Everything else waits until nobody talks.

Deterministic and clock-injectable, so it is unit-tested without a camera.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from scene import (
    DANGER_OBJECTS,
    FAST_MOVERS,
    PROXIMITY_ORDER,
    describe_distance,
    direction_hint,
    drop_off_warning,
    find_in_scene,
    nominal_distance,
    target_update,
)

MODES = ("quiet", "normal", "verbose")

# Proximity buckets that count as "in the way" for announcements.
ANNOUNCE_PROXIMITIES = {"very close", "close", "medium"}
# Side objects worth a word in verbose mode.
SIDE_PROXIMITIES = {"very close", "close"}

_STOP_KEY = "__stop__"
_DROP_KEY = "__drop__"
_DEPTH_KEY = "__depth__"

# Closing speed (m/s) that is clearly faster than the user walking toward a
# static object (~1.3 m/s): a cyclist, a car, someone running.
APPROACH_SPEED_MPS = 2.0

# Lower is more urgent; ties broken by proximity rank.
_KIND_PRIORITY = {
    "drop_off": 0,
    "stop": 1,
    "approaching": 2,
    "obstacle": 3,
    "closer": 3,
    "side": 5,
    "open": 6,
    "target": 7,
    "target_reached": 7,
    "target_lost": 8,
}


@dataclass
class Announcement:
    text: str
    urgent: bool  # may interrupt the agent's own speech
    kind: str = "obstacle"


@dataclass
class _Track:
    rank: int  # current proximity rank (0 = very close)
    seen_streak: int = 0
    missing_streak: int = 0
    announced_rank: int | None = None  # closest rank already announced
    announced_at: float = 0.0


@dataclass
class _Target:
    name: str
    labels: set[str]
    last_said: float = -1e9
    last_bucket: tuple | None = None
    missing: int = 0
    lost_said: float = -1e9
    seen_once: bool = False


class SceneMemory:
    def __init__(
        self,
        *,
        confirm_frames: int = 2,
        forget_after: int = 4,
        min_repeat_s: float = 4.0,
        target_interval_s: float = 3.0,
        mode: str = "normal",
        clock=time.monotonic,
    ) -> None:
        self.confirm_frames = confirm_frames  # debounce flickering detections
        self.forget_after = forget_after
        self.min_repeat_s = min_repeat_s
        self.target_interval_s = target_interval_s
        self.mode = mode
        self.paused = False
        self._clock = clock
        self._tracks: dict[str, _Track] = {}
        self._history: dict[tuple[str, str], deque] = {}
        self._approach_said: dict[str, float] = {}
        self._stop_said = False
        self._open_streak = 0
        self._target: _Target | None = None
        self._target_bucket: tuple | None = None

    # ── Controls ────────────────────────────────────────────────────
    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; use one of {MODES}")
        self.mode = mode

    def set_target(self, name: str, labels: set[str]) -> None:
        self._target = _Target(name=name, labels=set(labels))

    def clear_target(self) -> str | None:
        name = self._target.name if self._target else None
        self._target = None
        return name

    @property
    def target(self) -> str | None:
        return self._target.name if self._target else None

    def reset(self) -> None:
        self._tracks.clear()
        self._history.clear()
        self._approach_said.clear()
        self._stop_said = False
        self._open_streak = 0

    # ── Main entry ──────────────────────────────────────────────────
    def update(
        self,
        scene: dict[str, Any],
        *,
        announce: bool = True,
        urgent_only: bool = False,
    ) -> Announcement | None:
        """Feed one scene; return what to say, or None to stay quiet.

        ``announce=False`` (someone is talking) tracks objects without
        speaking; ``urgent_only=True`` (the agent is busy) lets only urgent
        warnings through. Either way, a change that isn't spoken stays
        pending and is announced on the first scene where it can be.
        """
        now = self._clock()
        path = scene.get("path") or {}
        ahead = self._in_path(scene)
        if path.get("drop_off"):
            ahead[_DROP_KEY] = {"proximity": "very close", "label": "drop"}
        if path.get("recommendation") == "stop":
            ahead[_STOP_KEY] = {"proximity": "very close", "label": "stop"}

        for key in list(self._tracks):
            if key not in ahead:
                track = self._tracks[key]
                track.seen_streak = 0
                track.missing_streak += 1
                if track.missing_streak >= self.forget_after:
                    del self._tracks[key]

        candidates: list[tuple[tuple, str, Announcement]] = []
        for key, obj in ahead.items():
            rank = _rank(obj)
            track = self._tracks.setdefault(key, _Track(rank=rank))
            track.rank = rank
            track.seen_streak += 1
            track.missing_streak = 0
            if track.seen_streak < self.confirm_frames:
                continue
            message = self._track_message(key, obj, track, path, now)
            if message is None:
                continue
            urgency = (
                _KIND_PRIORITY[message.kind],
                rank,
                obj.get("label") not in DANGER_OBJECTS,
            )
            candidates.append((urgency, key, message))

        candidates += self._approaching(scene, now)
        candidates += self._path_open(path)
        candidates += self._target_guidance(scene, now)

        stopping = _STOP_KEY in ahead
        if stopping and len(candidates) > 1:
            # An obstacle message already carries the "Stop" hint.
            candidates = [c for c in candidates if c[1] != _STOP_KEY]
        if self.mode == "quiet":
            candidates = [
                c for c in candidates if c[2].urgent or c[2].kind.startswith("target")
            ]
        if not candidates or not announce or self.paused:
            return None
        # Speak only the most urgent change; the rest are picked up next frame.
        urgency, key, message = min(candidates, key=lambda c: c[0])
        if urgent_only and not message.urgent:
            return None
        self._mark_said(key, message, now, stopping)
        return message

    # ── Pieces ──────────────────────────────────────────────────────
    def _in_path(self, scene: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Nearest in-path object per identity. Objects aren't individually
        tracked: the label is the identity, which is enough for guidance.
        Depth-only obstacles share one identity, so a surface that flickers
        between "wall" and "obstacle" isn't announced twice."""
        ahead: dict[str, dict[str, Any]] = {}
        for obj in scene.get("objects", []):
            if obj["position"] == "center":
                if obj["proximity"] not in ANNOUNCE_PROXIMITIES:
                    continue
                key = _DEPTH_KEY if obj.get("source") == "depth" else obj["label"]
            elif self.mode == "verbose" and obj["proximity"] in SIDE_PROXIMITIES:
                if obj.get("source") == "depth":
                    continue
                key = f"{obj['label']}@{obj['position']}"
            else:
                continue
            best = ahead.get(key)
            if best is None or _rank(obj) < _rank(best):
                ahead[key] = obj
        return ahead

    def _track_message(
        self, key: str, obj: dict, track: _Track, path: dict, now: float
    ) -> Announcement | None:
        if key == _DROP_KEY:
            if track.announced_rank is not None:
                return None
            return Announcement(drop_off_warning(path), urgent=True, kind="drop_off")
        if key == _STOP_KEY:
            if track.announced_rank is not None:
                return None
            return Announcement(direction_hint({}, path), urgent=True, kind="stop")
        if "@" in key:  # side object (verbose mode)
            if track.announced_rank is not None and track.rank >= track.announced_rank:
                return None
            text = (
                f"A {obj['label']} on your {obj['position']}, {describe_distance(obj)}."
            )
            return Announcement(text, urgent=False, kind="side")
        if track.announced_rank is None:
            return _new_obstacle(obj, path)
        if track.rank < track.announced_rank:
            if now - track.announced_at < self.min_repeat_s and track.rank > 0:
                return None  # recently spoken; only "very close" overrides
            return _closer_obstacle(obj, path)
        return None  # unchanged or moving away: stay silent

    def _approaching(self, scene: dict, now: float) -> list:
        """Objects closing in much faster than walking pace."""
        out = []
        seen: set[tuple[str, str]] = set()
        for obj in scene.get("objects", []):
            label = obj["label"]
            if label not in FAST_MOVERS or obj.get("distance_m") is None:
                continue
            key = (label, obj["position"])
            if key in seen:
                continue  # nearest instance per label+side only
            seen.add(key)
            hist = self._history.setdefault(key, deque(maxlen=8))
            hist.append((now, float(obj["distance_m"])))
            while hist and now - hist[0][0] > 2.0:
                hist.popleft()
            if len(hist) < 3:
                continue
            (t0, d0), (t1, d1) = hist[0], hist[-1]
            span = t1 - t0
            if span < 0.5:
                continue
            closing = d0 - d1
            speed = closing / span
            if speed < APPROACH_SPEED_MPS or closing < 1.0 or d1 > 4.0:
                continue
            if now - self._approach_said.get(label, -1e9) < 5.0:
                continue
            side = {
                "left": "from your left",
                "right": "from your right",
                "center": "from straight ahead",
            }[obj["position"]]
            text = f"Careful, a {label} is coming toward you fast {side}."
            urgency = (_KIND_PRIORITY["approaching"], _rank(obj), False)
            out.append(
                (urgency, f"approach:{label}", Announcement(text, True, "approaching"))
            )
        return out

    def _path_open(self, path: dict) -> list:
        if not self._stop_said:
            return []
        if path.get("recommendation") != "straight":
            self._open_streak = 0
            return []
        self._open_streak += 1
        if self._open_streak < max(2, self.confirm_frames):
            return []  # one open frame could be noise
        msg = Announcement("The way ahead looks open now.", False, "open")
        return [((_KIND_PRIORITY["open"], 3, True), "open", msg)]

    def _target_guidance(self, scene: dict, now: float) -> list:
        t = self._target
        if t is None:
            return []
        matches = find_in_scene(scene, t.labels)
        if not matches:
            t.missing += 1
            if t.missing >= self.forget_after and now - t.lost_said >= 6.0:
                if t.seen_once:
                    text = (
                        f"I can't see the {t.name} right now. Turn slowly to your "
                        "left or right so I can look for it."
                    )
                else:
                    text = (
                        f"I don't see a {t.name} in view yet. Turn slowly so I can "
                        "look for it, or say stop guidance."
                    )
                msg = Announcement(text, False, "target_lost")
                return [((_KIND_PRIORITY["target_lost"], 3, True), "target", msg)]
            return []
        t.missing = 0
        t.seen_once = True
        best = matches[0]
        if best["position"] == "center" and nominal_distance(best) < 1.0:
            text = (
                f"You've reached the {t.name}. It's right in front of you, "
                "within arm's reach."
            )
            msg = Announcement(text, False, "target_reached")
            return [((_KIND_PRIORITY["target_reached"], 0, True), "target", msg)]
        bucket = (best["position"], best["proximity"])
        due = now - t.last_said >= self.target_interval_s
        changed = bucket != t.last_bucket and now - t.last_said >= 1.0
        if not (due or changed):
            return []
        self._target_bucket = bucket
        msg = Announcement(target_update(t.name, best), False, "target")
        return [((_KIND_PRIORITY["target"], _rank(best), True), "target", msg)]

    def _mark_said(self, key: str, message: Announcement, now: float, stopping):
        if message.kind in ("stop", "drop_off") or (
            stopping and key in self._tracks and key != _STOP_KEY
        ):
            self._stop_said = True
            self._open_streak = 0
            if _STOP_KEY in self._tracks:
                self._tracks[_STOP_KEY].announced_rank = 0  # said as the hint
        if message.kind == "open":
            self._stop_said = False
        elif message.kind == "approaching":
            self._approach_said[key.split(":", 1)[1]] = now
        elif message.kind == "target_reached":
            self._target = None
        elif message.kind == "target_lost" and self._target:
            self._target.lost_said = now
        elif message.kind == "target" and self._target:
            # Remember what was said so a direction change speaks sooner.
            self._target.last_said = now
            self._target.last_bucket = self._target_bucket
        if key in self._tracks:
            track = self._tracks[key]
            track.announced_rank = track.rank
            track.announced_at = now


def _rank(obj: dict[str, Any]) -> int:
    return PROXIMITY_ORDER.index(obj["proximity"])


def _name(obj: dict[str, Any]) -> str:
    return "a wall" if obj["label"] == "wall" else f"a {obj['label']}"


def _with_hint(text: str, obj: dict[str, Any], path) -> str:
    hint = direction_hint(obj, path)
    return f"{text} {hint}" if hint else text


def _new_obstacle(obj: dict[str, Any], path) -> Announcement:
    if obj["proximity"] == "very close":
        text = f"Careful, {_name(obj)} directly ahead, {describe_distance(obj)}."
        return Announcement(_with_hint(text, obj, path), urgent=True, kind="obstacle")
    text = f"Heads up, {_name(obj)} ahead, {describe_distance(obj)}."
    return Announcement(_with_hint(text, obj, path), urgent=False, kind="obstacle")


def _closer_obstacle(obj: dict[str, Any], path) -> Announcement:
    the = "the wall" if obj["label"] == "wall" else f"the {obj['label']}"
    if obj["proximity"] == "very close":
        text = f"Careful, {the} ahead is very close."
        return Announcement(_with_hint(text, obj, path), urgent=True, kind="closer")
    text = f"You're getting closer to {the} ahead, {describe_distance(obj)}."
    return Announcement(_with_hint(text, obj, path), urgent=False, kind="closer")
