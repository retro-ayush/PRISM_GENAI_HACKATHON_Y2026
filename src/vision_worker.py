"""VisionGuide inference process: YOLO26 detection + YOLO26 depth.

Everything heavy (PyTorch, CUDA/MPS, Ultralytics, model weights) lives here and
is imported only inside this process. The voice agent starts it with the
multiprocessing "spawn" method, so it gets a fresh interpreter and its own
GPU context, and loading or inference can never block the LiveKit event loop.

IPC protocol (both queues carry plain picklable dicts):
  frame queue  -> {"frame_id", "sent_at", "jpeg": bytes} | None (stop)
  result queue <- {"type": "ready", "load_s", "device", "depth"}
                  {"type": "scene", "scene": {...}, "overlay": bytes | None,
                   "sent_at", "received_at"}
                  {"type": "error", "error"}

``overlay`` is the camera frame annotated with what the agent sees (boxes,
distances, free space, the direction arrow); the agent publishes it back into
the room as a video track so a sighted helper or a judge can follow along.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path
from typing import Any

import numpy as np

from scene import PATH_BLOCK_M, SIDES, RawDetection, build_scene

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _load_yolo(name: str):
    from ultralytics import YOLO

    path = Path(name)
    if not path.is_absolute():
        path = MODELS_DIR / name
    if path.exists():
        return YOLO(str(path))
    # Unknown locally: let Ultralytics download the release asset into models/.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(MODELS_DIR)
    try:
        return YOLO(path.name)
    finally:
        os.chdir(cwd)


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class VisionEngine:
    """Loads the models once and turns BGR frames into structured scenes."""

    def __init__(
        self,
        detect_model: str = "yolo26n.pt",
        depth_model: str | None = "yolo26n-depth.pt",
        device: str = "auto",
        min_confidence: float = 0.45,
        imgsz: int = 640,
    ) -> None:
        self.device = pick_device(device)
        self.min_confidence = min_confidence
        self.imgsz = imgsz
        self.detector = _load_yolo(detect_model)
        self.depth = _load_yolo(depth_model) if depth_model else None
        self._frames = 0
        # The first inference initialises kernels and can take seconds; do it
        # now so the first real frame is fast. Fall back to CPU if the chosen
        # accelerator cannot run the model.
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            self.process(dummy)
        except Exception:
            if self.device == "cpu":
                raise
            self.device = "cpu"
            self.process(dummy)

    def process(self, frame_bgr: np.ndarray, frame_id: int | None = None) -> dict:
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        result = self.detector(
            frame_bgr,
            verbose=False,
            device=self.device,
            conf=self.min_confidence,
            imgsz=self.imgsz,
        )[0]
        t1 = time.perf_counter()
        depth_map = None
        if self.depth is not None:
            depth_result = self.depth(
                frame_bgr, verbose=False, device=self.device, imgsz=self.imgsz
            )[0]
            depth_map = depth_result.depth.data.float().cpu().numpy()
        t2 = time.perf_counter()

        detections = []
        if result.boxes is not None:
            for box, conf, cls in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
                strict=True,
            ):
                detections.append(
                    RawDetection(
                        label=result.names[int(cls)],
                        confidence=float(conf),
                        box=tuple(box),
                    )
                )
        self._frames += 1
        return build_scene(
            detections,
            w,
            h,
            depth_map,
            min_confidence=self.min_confidence,
            frame_id=frame_id,
            performance={
                "detection_ms": round((t1 - t0) * 1000, 1),
                "depth_ms": round((t2 - t1) * 1000, 1),
                "total_ms": round((t2 - t0) * 1000, 1),
            },
        )


# ── Overlay ──────────────────────────────────────────────────────────

_RED = (0, 0, 255)
_ORANGE = (0, 165, 255)
_YELLOW = (0, 220, 255)
_GREEN = (0, 200, 0)
_WHITE = (255, 255, 255)
_GREY = (150, 150, 150)


def _clearance_color(c: float | None) -> tuple[int, int, int]:
    if c is None:
        return _GREY
    if c < PATH_BLOCK_M:
        return _RED
    if c < 2 * PATH_BLOCK_M:
        return _YELLOW
    return _GREEN


def _wrap(text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines[:2]


def draw_overlay(frame_bgr: np.ndarray, scene: dict[str, Any]) -> np.ndarray:
    """Annotate a frame with what the agent sees. Pure OpenCV, ~1-2 ms."""
    import cv2

    out = frame_bgr.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    path = scene.get("path") or {}
    clearance = path.get("clearance_m") or {}

    # Columns and their free space.
    for x in (w // 3, 2 * w // 3):
        cv2.line(out, (x, 0), (x, h), (90, 90, 90), 1)
    for i, side in enumerate(SIDES):
        c = clearance.get(side)
        label = (
            f"{side[0].upper()} {c:.1f}m" if c is not None else f"{side[0].upper()} ?"
        )
        cv2.putText(
            out, label, (w * i // 3 + 8, 26), font, 0.65, _clearance_color(c), 2
        )

    # Depth-only obstacles (walls, boxes): tint the column band they block.
    for o in scene.get("objects", []):
        if o.get("source") != "depth":
            continue
        i = SIDES.index(o["position"])
        x0, x1 = w * i // 3, w * (i + 1) // 3
        tint = out.copy()
        cv2.rectangle(tint, (x0, int(h * 0.25)), (x1, int(h * 0.70)), _RED, -1)
        cv2.addWeighted(tint, 0.25, out, 0.75, 0, out)
        cv2.putText(
            out,
            f"{o['label'].upper()} {o['distance_m']:.1f}m",
            (x0 + 8, int(h * 0.25) + 24),
            font,
            0.6,
            _RED,
            2,
        )

    # Detected objects.
    for o in scene.get("objects", []) + scene.get("others", []):
        box = o.get("box")
        if not box:
            continue
        x1, y1, x2, y2 = box
        navigation = o in scene.get("objects", [])
        if o["position"] == "center" and o["proximity"] in ("very close", "close"):
            color = _RED
        elif o["position"] == "center" and navigation:
            color = _ORANGE
        else:
            color = _WHITE if navigation else _GREY
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        dist = o.get("distance_m")
        text = f"{o['label']} {dist:.1f}m" if dist is not None else f"{o['label']}"
        if o.get("pass_on"):
            text += f" > pass {o['pass_on']}"
        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
        # Above the box, unless that would collide with the clearance row.
        ty = y1 - 6 if y1 > 44 else y1 + th + 10
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (0, 0, 0), -1)
        cv2.putText(out, text, (x1 + 3, ty - 2), font, 0.5, color, 1)

    # Drop-off cue.
    if path.get("drop_off"):
        y = int(h * 0.82)
        cv2.rectangle(out, (w // 3, y - 22), (2 * w // 3, y + 8), _RED, -1)
        cv2.putText(
            out,
            f"DROP-OFF? {path['drop_off']['distance_m']:.1f}m",
            (w // 3 + 10, y),
            font,
            0.6,
            _WHITE,
            2,
        )

    # Direction arrow (bottom centre) and guidance text.
    rec = path.get("recommendation", "straight")
    cx, cy = w // 2, h - 70
    if rec == "straight":
        cv2.arrowedLine(out, (cx, cy + 20), (cx, cy - 20), _GREEN, 4, tipLength=0.4)
    elif rec == "step_left":
        cv2.arrowedLine(out, (cx + 24, cy), (cx - 24, cy), _YELLOW, 4, tipLength=0.4)
    elif rec == "step_right":
        cv2.arrowedLine(out, (cx - 24, cy), (cx + 24, cy), _YELLOW, 4, tipLength=0.4)
    else:
        cv2.rectangle(out, (cx - 40, cy - 18), (cx + 40, cy + 14), _RED, -1)
        cv2.putText(out, "STOP", (cx - 30, cy + 6), font, 0.7, _WHITE, 2)

    lines = _wrap(scene.get("guidance", ""), max(30, w // 11))
    bar_h = 16 + 20 * len(lines)
    bar = out.copy()
    cv2.rectangle(bar, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.6, out, 0.4, 0, out)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, h - bar_h + 20 + 20 * i), font, 0.52, _WHITE, 1)

    perf = scene.get("performance") or {}
    if perf.get("total_ms") is not None:
        text = f"{perf['total_ms']:.0f} ms"
        (tw, _), _ = cv2.getTextSize(text, font, 0.55, 2)
        cv2.putText(out, text, (w - tw - 10, 26), font, 0.55, _WHITE, 2)
    return out


def encode_overlay(frame_bgr: np.ndarray, scene: dict[str, Any]) -> bytes:
    import cv2

    ok, buf = cv2.imencode(
        ".jpg", draw_overlay(frame_bgr, scene), [cv2.IMWRITE_JPEG_QUALITY, 70]
    )
    return buf.tobytes() if ok else b""


# ── Process loop ─────────────────────────────────────────────────────


def _put_latest(q: Any, item: Any) -> None:
    """Put without blocking; if the queue is full, drop the oldest item."""
    try:
        q.put_nowait(item)
    except queue.Full:
        with contextlib.suppress(queue.Empty):
            q.get_nowait()
        with contextlib.suppress(queue.Full):
            q.put_nowait(item)


def _serve(frame_q: Any, result_q: Any, process_frame) -> None:
    parent = mp.parent_process()
    while True:
        try:
            item = frame_q.get(timeout=1.0)
        except queue.Empty:
            if parent is not None and not parent.is_alive():
                return  # the agent died; don't linger holding the GPU
            continue
        if item is None:
            return
        received_at = time.time()
        try:
            scene, overlay = process_frame(item)
        except Exception as e:  # keep serving: one bad frame isn't fatal
            _put_latest(result_q, {"type": "error", "error": repr(e)})
            continue
        _put_latest(
            result_q,
            {
                "type": "scene",
                "scene": scene,
                "overlay": overlay,
                "sent_at": item["sent_at"],
                "received_at": received_at,
            },
        )


def run_worker(frame_q: Any, result_q: Any, config: dict[str, Any]) -> None:
    """Process entry point (must be module-level so "spawn" can import it)."""
    import cv2

    t0 = time.perf_counter()
    try:
        engine = VisionEngine(
            detect_model=config.get("detect_model", "yolo26n.pt"),
            depth_model=config.get("depth_model", "yolo26n-depth.pt") or None,
            device=config.get("device", "auto"),
            min_confidence=float(config.get("min_confidence", 0.45)),
        )
    except Exception as e:
        result_q.put({"type": "error", "error": f"model load failed: {e!r}"})
        return
    result_q.put(
        {
            "type": "ready",
            "load_s": round(time.perf_counter() - t0, 2),
            "device": engine.device,
            "depth": engine.depth is not None,
        }
    )
    overlay_on = bool(config.get("overlay", True))

    def process_frame(item: dict) -> tuple[dict, bytes | None]:
        buf = np.frombuffer(item["jpeg"], dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("could not decode frame")
        scene = engine.process(frame, frame_id=item["frame_id"])
        return scene, encode_overlay(frame, scene) if overlay_on else None

    _serve(frame_q, result_q, process_frame)


def run_fake_worker(frame_q: Any, result_q: Any, config: dict[str, Any]) -> None:
    """Torch-free stand-in for tests and demos without a GPU.

    Reports a chair straight ahead that gets closer with every frame.
    """
    time.sleep(float(config.get("fake_load_s", 0)))
    result_q.put({"type": "ready", "load_s": 0.0, "device": "fake", "depth": True})
    overlay_on = bool(config.get("overlay", False))

    def process_frame(item: dict) -> tuple[dict, bytes | None]:
        fid = item["frame_id"]
        distance = max(0.5, 4.5 - 0.5 * fid)
        # A 300x400 box in a 640x480 frame; depth constant inside it.
        depth = np.full((480, 640), 10.0, dtype=np.float32)
        depth[40:440, 170:470] = distance
        det = RawDetection("chair", 0.9, (170.0, 40.0, 470.0, 440.0))
        scene = build_scene(
            [det],
            640,
            480,
            depth,
            frame_id=fid,
            performance={"detection_ms": 1.0, "depth_ms": 1.0, "total_ms": 2.0},
        )
        overlay = None
        if overlay_on:
            canvas = np.full((480, 640, 3), 40, dtype=np.uint8)
            overlay = encode_overlay(canvas, scene)
        return scene, overlay

    _serve(frame_q, result_q, process_frame)
