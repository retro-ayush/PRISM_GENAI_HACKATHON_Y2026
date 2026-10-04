"""VisionGuide scene representation: the bridge between computer vision and
the voice agent.

Pure functions only (NumPy, no torch), so the agent process can import this
module without pulling in the heavy vision stack, and it can be unit-tested
without a camera or GPU. The vision worker turns raw detections plus a depth
map into a structured scene; the LLM only ever sees that structured scene.

A scene has:
- ``objects``: navigation-relevant detections, most urgent first, each with
  position (left/center/right), clock bearing, elevation, approximate
  distance and which side to pass it on;
- ``others``: everything else the detector saw (for "where is my phone?");
- ``path``: free space per column, wall and drop-off detection from the
  depth map, and a walking recommendation;
- ``guidance``: one or two spoken sentences, the most urgent thing first.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

# Objects that matter for moving around; everything else goes to ``others``.
NAVIGATION_OBJECTS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "cat",
    "chair",
    "couch",
    "bench",
    "bed",
    "dining table",
    "potted plant",
    "suitcase",
    "backpack",
    "fire hydrant",
    "stop sign",
    "traffic light",
    "bottle",
    "cell phone",
}

# Moving or large objects get priority over static clutter at equal range.
DANGER_OBJECTS = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "dog"}
# Objects that can close in fast enough to matter (see SceneMemory).
FAST_MOVERS = {"bicycle", "motorcycle", "car", "bus", "truck", "person", "dog"}

PROXIMITY_ORDER = ["very close", "close", "medium", "far"]
SIDES = ("left", "center", "right")

# Anything nearer than this in front of the user blocks the path (metres).
PATH_BLOCK_M = 1.5
# Assumed horizontal field of view of a phone/webcam, for clock bearings.
DEFAULT_HFOV_DEG = 66.0

# Rough distance per proximity bucket, for free-space checks without depth.
_NOMINAL_DISTANCE = {"very close": 0.7, "close": 1.2, "medium": 3.0, "far": 6.0}
_ADVICE = {
    "straight": None,
    "step_left": "Move to your left.",
    "step_right": "Move to your right.",
    "stop": "Stop. The way ahead looks blocked on all sides.",
    "stop_drop": "Stop. The ground ahead may drop away.",
}

# Metric thresholds (metres) for proximity buckets when depth is available.
_DISTANCE_BUCKETS = ((1.0, "very close"), (2.0, "close"), (4.0, "medium"))
# Bounding-box area ratio thresholds, used only when depth is unavailable.
_AREA_BUCKETS = ((0.18, "very close"), (0.07, "close"), (0.025, "medium"))

# The 80 COCO classes YOLO26 detects, for resolving what the user asks for.
COCO_LABELS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
}  # fmt: skip

# What people say -> what the detector calls it.
OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "phone": ("cell phone",),
    "mobile": ("cell phone",),
    "cellphone": ("cell phone",),
    "smartphone": ("cell phone",),
    "bag": ("backpack", "handbag", "suitcase"),
    "purse": ("handbag",),
    "luggage": ("suitcase",),
    "seat": ("chair", "couch", "bench"),
    "sofa": ("couch",),
    "table": ("dining table",),
    "desk": ("dining table",),
    "plant": ("potted plant",),
    "television": ("tv",),
    "screen": ("tv", "laptop"),
    "monitor": ("tv",),
    "computer": ("laptop",),
    "water": ("bottle",),
    "drink": ("bottle", "cup", "wine glass"),
    "mug": ("cup",),
    "glass": ("wine glass", "cup"),
    "people": ("person",),
    "someone": ("person",),
    "anyone": ("person",),
    "man": ("person",),
    "woman": ("person",),
    "child": ("person",),
    "kid": ("person",),
    "bike": ("bicycle",),
    "cycle": ("bicycle",),
    "motorbike": ("motorcycle",),
    "scooter": ("motorcycle",),
    "vehicle": ("car", "bus", "truck", "motorcycle"),
    "fridge": ("refrigerator",),
    "pet": ("dog", "cat"),
    "puppy": ("dog",),
    "kitten": ("cat",),
    "ball": ("sports ball",),
    "racket": ("tennis racket",),
    "hydrant": ("fire hydrant",),
    "signal": ("traffic light",),
    "light": ("traffic light",),
    "sign": ("stop sign",),
    "hairdryer": ("hair drier",),
    "teddy": ("teddy bear",),
    "toy": ("teddy bear", "sports ball", "kite"),
    "food": ("banana", "apple", "sandwich", "orange", "pizza", "donut", "cake"),
    "fruit": ("banana", "apple", "orange"),
    "cutlery": ("fork", "knife", "spoon"),
}
_STOP_WORDS = {"my", "the", "a", "an", "some", "any", "your", "this", "that", "is"}


@dataclass
class RawDetection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


# ── Geometry helpers ─────────────────────────────────────────────────


def classify_position(center_x: float, frame_width: int) -> str:
    if center_x < frame_width / 3:
        return "left"
    if center_x > frame_width * 2 / 3:
        return "right"
    return "center"


def classify_elevation(center_y: float, frame_height: int) -> str:
    """Where the object sits vertically: high (above eye level), level, low."""
    if center_y < frame_height / 3:
        return "high"
    if center_y > frame_height * 2 / 3:
        return "low"
    return "level"


def bearing_deg(center_x: float, frame_width: int, hfov: float = DEFAULT_HFOV_DEG):
    """Horizontal angle from straight ahead; negative is left."""
    return round((center_x / frame_width - 0.5) * hfov, 1)


def clock_position(bearing: float) -> str:
    """Orientation-and-mobility clock face: 12 is straight ahead, 9 is left."""
    hour = round(bearing / 30.0)
    hour = (12 + hour - 1) % 12 + 1
    return f"{hour} o'clock"


def proximity_from_distance(distance_m: float) -> str:
    for limit, bucket in _DISTANCE_BUCKETS:
        if distance_m < limit:
            return bucket
    return "far"


def proximity_from_area(box_area: float, frame_area: float) -> str:
    ratio = box_area / frame_area if frame_area else 0.0
    for limit, bucket in _AREA_BUCKETS:
        if ratio > limit:
            return bucket
    return "far"


def object_distance(depth: np.ndarray, box: tuple[float, float, float, float]):
    """Robust distance (m) to an object: median depth of the box's inner half.

    The inner crop avoids background pixels around the object's edges; the
    median ignores depth-map outliers. Returns None if nothing valid is left.
    """
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    xa, xb = int(max(0, x1 + bw / 4)), int(min(w, x2 - bw / 4))
    ya, yb = int(max(0, y1 + bh / 4)), int(min(h, y2 - bh / 4))
    if xb <= xa or yb <= ya:
        return None
    region = depth[ya:yb, xa:xb]
    region = region[np.isfinite(region) & (region > 0)]
    if region.size == 0:
        return None
    return float(np.median(region))


def nominal_distance(obj: dict[str, Any]) -> float:
    """Distance in metres, or a typical value for the proximity bucket."""
    d = obj.get("distance_m")
    return float(d) if d is not None else _NOMINAL_DISTANCE[obj["proximity"]]


def _priority(obj: dict[str, Any]) -> tuple:
    return (
        obj["position"] != "center",
        PROXIMITY_ORDER.index(obj["proximity"]),
        obj["label"] not in DANGER_OBJECTS | {"wall", "obstacle"},
        nominal_distance(obj),
    )


# ── Free space, walls, drop-offs ─────────────────────────────────────


def analyze_path(
    depth: np.ndarray | None,
    objects: list[dict[str, Any]],
    *,
    block_m: float = PATH_BLOCK_M,
) -> dict[str, Any]:
    """Free space ahead, left and right, hazards, and which way to go.

    With depth, clearance per column is a low percentile of the depth map in
    the middle band of the image (rows 25-70%). That band is roughly where
    obstacles at body height appear for a forward-facing camera, while the
    floor near the feet (bottom rows) is excluded so it doesn't read as an
    obstacle. The map is downsampled 4x first: this costs well under 1 ms.

    Without depth, clearance falls back to the detected objects' proximity.
    """
    clearance: dict[str, float | None] = {}
    wall_ahead = False
    drop_off = None
    if depth is not None:
        d = depth[::4, ::4]
        h, w = d.shape[:2]
        band = d[int(h * 0.25) : int(h * 0.70)]
        for i, side in enumerate(SIDES):
            col = band[:, int(w * i / 3) : int(w * (i + 1) / 3)]
            valid = col[np.isfinite(col) & (col > 0)]
            clearance[side] = (
                round(float(np.percentile(valid, 15)), 1) if valid.size else None
            )
        wall_ahead = _looks_like_wall(d, clearance.get("center"), block_m)
        edge = detect_drop_off(d)
        if edge is not None:
            drop_off = {"distance_m": edge}
    else:
        for side in SIDES:
            near = [nominal_distance(o) for o in objects if o["position"] == side]
            clearance[side] = min(near) if near else None  # None = nothing seen

    def is_open(side: str) -> bool:
        c = clearance[side]
        return c is None or c >= block_m

    if drop_off is not None:
        recommendation = "stop_drop"
    elif is_open("center"):
        recommendation = "straight"
    else:
        sides = [s for s in ("left", "right") if is_open(s)]
        if not sides:
            recommendation = "stop"
        else:
            # Prefer the side with more room; unknown (None) counts as open.
            recommendation = "step_" + max(
                sides, key=lambda s: clearance[s] if clearance[s] is not None else 99
            )
    return {
        "clearance_m": clearance,
        "wall_ahead": wall_ahead,
        "drop_off": drop_off,
        "recommendation": recommendation,
        "advice": _ADVICE[recommendation],
    }


def _looks_like_wall(d: np.ndarray, center_clearance, block_m: float) -> bool:
    """Something near that fills the center and is equally far top to bottom.

    A wall (or door, cupboard, large panel) is a near, vertical surface: depth
    in the center column is roughly constant across rows 15-75%. A floor gets
    steadily closer toward the bottom, and a person or chair fills only part
    of the column, so neither passes both tests.
    """
    if center_clearance is None or center_clearance >= block_m:
        return False
    h, w = d.shape[:2]
    col = d[int(h * 0.15) : int(h * 0.75), int(w / 3) : int(w * 2 / 3)]
    near_fraction = float(np.mean(col < center_clearance + 0.5))
    rows = np.median(col, axis=1)
    spread = float((rows.max() - rows.min()) / max(float(np.median(rows)), 1e-3))
    return near_fraction > 0.6 and spread < 0.35


def detect_drop_off(
    d: np.ndarray,
    *,
    max_edge_m: float = 3.0,
    jump_ratio: float = 1.8,
    min_jump_m: float = 0.8,
) -> float | None:
    """Distance (m) to a likely drop in the ground ahead, or None.

    Scans the lower half of the center column from the bottom up in bands.
    The floor recedes smoothly (each band a little farther than the one
    below). A kerb edge, platform edge or downward stairs shows up as the
    floor suddenly jumping far away where the ground falls out of view.
    An obstacle's top edge also jumps to the background, but the bands below
    it are flat (a vertical surface), not receding, so it is not reported.
    Experimental: monocular depth is approximate, so this is a cue, not a
    measurement, and the agent says so.
    """
    h, w = d.shape[:2]
    col = d[int(h * 0.5) :, int(w / 3) : int(w * 2 / 3)]
    rows = col.shape[0]
    band_h = max(2, rows // 10)
    bands: list[float] = []
    for top in range(rows - band_h, -1, -band_h):  # bottom band first
        chunk = col[top : top + band_h]
        valid = chunk[np.isfinite(chunk) & (chunk > 0)]
        if valid.size == 0:
            return None
        bands.append(float(np.median(valid)))
    if len(bands) < 4:
        return None
    for i in range(2, len(bands) - 1):  # skip the very bottom band (feet, cane)
        near, far = bands[i], bands[i + 1]
        if near > max_edge_m or far / near < jump_ratio or far - near < min_jump_m:
            continue
        receding = bands[i] > bands[i - 1] * 1.03 and bands[i] >= bands[i - 2] * 1.08
        if receding:
            return round(near, 1)
    return None


def _pass_side(
    obj: dict[str, Any],
    clearance: dict,
    center_x: float,
    width: int,
    objects: list[dict[str, Any]],
) -> str:
    """Which side to walk around an obstacle ahead.

    1. Avoid a side with another detected object at a similar or nearer range.
    2. Otherwise prefer the side with clearly more free space.
    3. Otherwise go away from where the object sits in the frame.
    """
    reach = nominal_distance(obj) + 1.0

    def occupied(side: str) -> bool:
        return any(
            o["position"] == side and nominal_distance(o) <= reach for o in objects
        )

    left_busy, right_busy = occupied("left"), occupied("right")
    if left_busy != right_busy:
        return "right" if left_busy else "left"
    left, right = clearance.get("left"), clearance.get("right")
    left = 99.0 if left is None else left
    right = 99.0 if right is None else right
    if abs(left - right) >= 0.5:
        return "left" if left > right else "right"
    return "right" if center_x < width / 2 else "left"


# ── Scene assembly ───────────────────────────────────────────────────


def build_scene(
    detections: list[RawDetection],
    frame_width: int,
    frame_height: int,
    depth: np.ndarray | None = None,
    *,
    min_confidence: float = 0.45,
    performance: dict[str, float] | None = None,
    frame_id: int | None = None,
    timestamp: float | None = None,
    hfov: float = DEFAULT_HFOV_DEG,
) -> dict[str, Any]:
    """Structured scene: objects with position, distance and proximity, plus
    free-space analysis and a walking direction.

    Objects are sorted most-urgent first (directly ahead, then nearest).
    Obstacles the detector can't name (walls, doors, boxes) are added from the
    depth map as "wall" or "obstacle".
    """
    frame_area = frame_width * frame_height
    objects: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    centers: list[float] = []
    for det in detections:
        if det.confidence < min_confidence:
            continue
        x1, y1, x2, y2 = det.box
        distance = object_distance(depth, det.box) if depth is not None else None
        if distance is not None:
            proximity = proximity_from_distance(distance)
        else:
            proximity = proximity_from_area((x2 - x1) * (y2 - y1), frame_area)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bearing = bearing_deg(cx, frame_width, hfov)
        obj = {
            "label": det.label,
            "confidence": round(det.confidence, 2),
            "position": classify_position(cx, frame_width),
            "clock": clock_position(bearing),
            "elevation": classify_elevation(cy, frame_height),
            "distance_m": round(distance, 1) if distance is not None else None,
            "proximity": proximity,
            "box": [round(v) for v in det.box],
        }
        if det.label in NAVIGATION_OBJECTS:
            objects.append(obj)
            centers.append(cx)
        else:
            others.append(obj)

    path = analyze_path(depth, objects)
    clearance = path["clearance_m"]
    for obj, cx in zip(objects, centers, strict=True):
        if obj["position"] == "center":
            obj["pass_on"] = _pass_side(obj, clearance, cx, frame_width, objects)
    # A named obstacle in the way (within ~4 m) sets the direction even when
    # the depth band still looks open: start going around it early.
    ahead = sorted(
        (o for o in objects if o["position"] == "center" and o["proximity"] != "far"),
        key=lambda o: PROXIMITY_ORDER.index(o["proximity"]),
    )
    if ahead and path["recommendation"] == "straight":
        path["recommendation"] = "step_" + ahead[0]["pass_on"]
        path["advice"] = _ADVICE[path["recommendation"]]

    if depth is not None:
        objects.extend(_unnamed_obstacles(objects, path))
    objects.sort(key=_priority)
    others.sort(key=nominal_distance)
    return {
        "timestamp": time.time() if timestamp is None else timestamp,
        "frame_id": frame_id,
        "objects": objects,
        "others": others,
        "path": path,
        "guidance": navigation_guidance(objects, path),
        "depth_available": depth is not None,
        "frame": {"width": frame_width, "height": frame_height},
        "performance": performance or {},
    }


_SIDE_CLOCK = {"left": "10 o'clock", "center": "12 o'clock", "right": "2 o'clock"}


def _unnamed_obstacles(objects: list[dict], path: dict) -> list[dict]:
    """Obstacles seen in the depth map that no detection explains."""
    out = []
    clearance = path["clearance_m"]
    for side in SIDES:
        c = clearance.get(side)
        # Ahead: anything blocking the path. Sides: only when very close
        # (e.g. about to brush a wall), to avoid chatter in corridors.
        limit = PATH_BLOCK_M if side == "center" else 1.0
        if c is None or c >= limit:
            continue
        explained = any(
            o["position"] == side
            and o["distance_m"] is not None
            and abs(o["distance_m"] - c) < 0.7
            for o in objects
        )
        if explained:
            continue
        obstacle = {
            "label": "wall" if side == "center" and path["wall_ahead"] else "obstacle",
            "confidence": None,
            "position": side,
            "clock": _SIDE_CLOCK[side],
            "elevation": "level",
            "distance_m": c,
            "proximity": proximity_from_distance(c),
            "source": "depth",
        }
        if side == "center" and path["recommendation"].startswith("step_"):
            obstacle["pass_on"] = path["recommendation"][5:]
        out.append(obstacle)
    return out


# ── Spoken phrasing ──────────────────────────────────────────────────


def describe_distance(obj: dict[str, Any]) -> str:
    """Spoken, deliberately approximate distance ("about 1.5 metres")."""
    d = obj.get("distance_m")
    if d is None:
        return obj["proximity"]
    if d < 1.0:
        return "less than a metre away"
    # Round to the nearest half metre: the depth model is only approximate.
    rounded = round(d * 2) / 2
    text = f"{rounded:g}"
    return f"about {text} metre{'' if rounded == 1 else 's'} away"


def spoken_location(obj: dict[str, Any], *, elevation: bool = False) -> str:
    """ "on your left, about 2 metres away, low down"."""
    side = {
        "left": "on your left",
        "center": "straight ahead",
        "right": "on your right",
    }[obj["position"]]
    text = f"{side}, {describe_distance(obj)}"
    if elevation and obj.get("elevation") == "low":
        text += ", low down"
    elif elevation and obj.get("elevation") == "high":
        text += ", up high"
    return text


def direction_hint(obj: dict[str, Any], path: dict[str, Any] | None) -> str:
    """Short spoken direction for an obstacle ahead ("Go to your right.")."""
    if path and path.get("recommendation") in ("stop", "stop_drop"):
        return _ADVICE[path["recommendation"]]
    side = obj.get("pass_on")
    return f"Go to your {side}." if side else ""


def drop_off_warning(path: dict[str, Any]) -> str:
    edge = path["drop_off"]["distance_m"]
    where = describe_distance({"distance_m": edge}).replace(" away", " ahead")
    return f"Careful: the ground may drop away {where}. Stop and check with your cane."


def navigation_guidance(
    objects: list[dict[str, Any]], path: dict[str, Any] | None = None
) -> str:
    """Spoken guidance, assuming objects are sorted by priority."""
    if path and path.get("drop_off"):
        return drop_off_warning(path)
    ahead = [o for o in objects if o["position"] == "center"]
    if ahead:
        o = ahead[0]
        what = "a wall" if o["label"] == "wall" else f"a {o['label']}"
        if o["proximity"] in ("very close", "close"):
            text = f"Warning: {what} directly ahead, {describe_distance(o)}."
        else:
            text = f"{what[0].upper()}{what[1:]} ahead, {describe_distance(o)}."
        hint = direction_hint(o, path)
        return f"{text} {hint}".strip()
    if not objects:
        return "No obstacles detected ahead."
    left = [o for o in objects if o["position"] == "left"]
    right = [o for o in objects if o["position"] == "right"]
    parts = [
        f"a {side[0]['label']} on your {name}, {describe_distance(side[0])}"
        for name, side in (("left", left), ("right", right))
        if side
    ]
    return "Nothing detected directly ahead; " + " and ".join(parts) + "."


# ── Finding things ───────────────────────────────────────────────────


def resolve_object_query(query: str) -> tuple[str, set[str]]:
    """Map what the user said ("my phone") to detector labels ({"cell phone"}).

    Returns the cleaned name and the matching labels (empty if unknown).
    """
    words = [w for w in re.findall(r"[a-z]+", query.lower()) if w not in _STOP_WORDS]
    name = " ".join(words)
    if not name:
        return "", set()
    candidates = [name]
    if name.endswith("es"):
        candidates.append(name[:-2])
    if name.endswith("s"):
        candidates.append(name[:-1])
    for cand in candidates:
        if cand in OBJECT_ALIASES:
            return name, set(OBJECT_ALIASES[cand])
        if cand in COCO_LABELS:
            return name, {cand}
    # Last word only ("red chair" -> chair).
    last = words[-1]
    for cand in (last, last[:-1] if last.endswith("s") else last):
        if cand in OBJECT_ALIASES:
            return name, set(OBJECT_ALIASES[cand])
        if cand in COCO_LABELS:
            return name, {cand}
    return name, set()


def find_in_scene(scene: dict[str, Any], labels: set[str]) -> list[dict[str, Any]]:
    """All detections with one of ``labels``, nearest first."""
    pool = list(scene.get("objects", [])) + list(scene.get("others", []))
    found = [o for o in pool if o.get("label") in labels and o.get("source") != "depth"]
    return sorted(found, key=nominal_distance)


def target_update(name: str, obj: dict[str, Any]) -> str:
    """One spoken guidance step toward a target the user asked to reach."""
    if obj["position"] == "center" and obj["proximity"] in ("very close", "close"):
        return f"The {name} is just ahead, {describe_distance(obj)}. Keep going slowly."
    turn = {
        "left": " Turn a little to your left.",
        "right": " Turn a little to your right.",
        "center": "",
    }[obj["position"]]
    return f"The {name} is {spoken_location(obj)}.{turn}"
