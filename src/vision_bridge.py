"""Agent-side half of VisionGuide: LiveKit camera -> vision process -> scene.

Runs inside the voice agent process and never imports torch. It

- reads the user's LiveKit camera track with ``rtc.VideoStream``,
- throttles, downsizes and JPEG-encodes frames off the event loop,
- sends them to a separately *spawned* vision process through a queue of
  size 1, so a busy worker always gets the newest frame and old frames are
  dropped instead of building a backlog,
- polls the result queue and keeps ``latest_scene`` in memory,
- keeps the latest raw frame for the vision-language model ("describe what
  you see", "read this sign"),
- hands the worker's annotated frame to an ``OverlayPublisher`` so what the
  agent sees can be streamed back into the room,
- restarts the worker if it crashes, and tracks vision latency metrics.

The worker process is only started when a camera track actually arrives, so
voice-only sessions (e.g. benchmark runs) never load the vision models.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import queue
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from vision_worker import _put_latest, run_worker

logger = logging.getLogger("vision")


class VisionBridge:
    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        target: Callable[..., None] = run_worker,
        max_fps: float = 8.0,
        max_width: int = 640,
        jpeg_quality: int = 80,
        scene_max_age_s: float = 5.0,
        max_restarts: int = 3,
        on_scene: Callable[[dict[str, Any]], None] | None = None,
        on_overlay: Callable[[bytes], None] | None = None,
    ) -> None:
        self.config = config or {}
        self.target = target
        self.max_fps = max_fps
        self.max_width = max_width
        self.jpeg_quality = jpeg_quality
        self.scene_max_age_s = scene_max_age_s
        self.max_restarts = max_restarts
        self.on_scene = on_scene
        self.on_overlay = on_overlay

        self.state = "idle"  # idle | starting | ready | failed | closed
        self.error: str | None = None
        self.device: str | None = None
        self.latest_scene: dict[str, Any] | None = None
        self.latest_jpeg: bytes | None = None  # newest raw camera frame
        self.latest_jpeg_at = 0.0
        self.has_camera = False

        self._ctx = mp.get_context("spawn")
        self._proc: Any = None
        self._frame_q: Any = None
        self._result_q: Any = None
        self._reader: asyncio.Task | None = None
        self._poller: asyncio.Task | None = None
        self._restarts = 0
        self._frame_ids = 0

        self.stats: dict[str, Any] = {
            "frames_sent": 0,
            "frames_dropped": 0,  # replaced in the queue before the worker took them
            "scenes": 0,
            "errors": 0,
            "restarts": 0,
            "load_s": None,
        }
        self._recent: deque[dict[str, float]] = deque(maxlen=50)

    # ── Lifecycle ───────────────────────────────────────────────────
    async def start(self) -> None:
        """Spawn the vision process (idempotent). Never blocks the loop."""
        if self.state in ("starting", "ready", "closed"):
            return
        self.state = "starting"
        self.error = None
        self._frame_q = self._ctx.Queue(maxsize=1)
        self._result_q = self._ctx.Queue(maxsize=4)
        self._proc = self._ctx.Process(
            target=self.target,
            args=(self._frame_q, self._result_q, self.config),
            name="visionguide-worker",
            daemon=True,
        )
        try:
            await asyncio.to_thread(self._proc.start)
        except Exception as e:
            logger.exception("could not start the vision process")
            self.state, self.error, self._proc = "failed", repr(e), None
            return
        logger.info("vision process started (pid %s)", self._proc.pid)
        if self._poller is None or self._poller.done():
            self._poller = asyncio.create_task(self._poll_results(), name="vision_poll")

    def attach_track(self, track: Any) -> None:
        """Start reading frames from a subscribed LiveKit video track."""
        if self.state == "closed":
            return
        if self._reader and not self._reader.done():
            self._reader.cancel()
        self.has_camera = True
        self._reader = asyncio.create_task(self._read_frames(track), name="vision_read")

    def detach_track(self) -> None:
        self.has_camera = False
        if self._reader and not self._reader.done():
            self._reader.cancel()
        self._reader = None

    async def aclose(self) -> None:
        self.state = "closed"
        self.detach_track()
        if self._poller:
            self._poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poller
        await asyncio.to_thread(self._stop_process)

    def _stop_process(self) -> None:
        if self._proc is None or self._proc.pid is None:
            self._proc = None
            return
        with contextlib.suppress(Exception):
            _put_latest(self._frame_q, None)
        self._proc.join(timeout=3)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2)
        self._proc = None

    # ── Frames in ───────────────────────────────────────────────────
    async def _read_frames(self, track: Any) -> None:
        from livekit import rtc

        stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)
        await self.start()
        min_interval = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
        last_sent = 0.0
        try:
            async for event in stream:
                now = time.monotonic()
                if now - last_sent < min_interval:
                    continue  # throttled
                last_sent = now
                frame = event.frame
                jpeg = await asyncio.to_thread(
                    self.encode_rgb, frame.data, frame.width, frame.height
                )
                self.latest_jpeg, self.latest_jpeg_at = jpeg, time.time()
                if self.state == "ready":  # worker still loading otherwise
                    self.submit_jpeg(jpeg)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("camera stream failed")
        finally:
            await stream.aclose()

    def encode_rgb(self, data: Any, width: int, height: int) -> bytes:
        """RGB24 buffer -> downsized BGR JPEG (runs in a worker thread)."""
        import cv2

        rgb = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        if width > self.max_width:
            scale = self.max_width / width
            rgb = cv2.resize(
                rgb, (self.max_width, int(height * scale)), interpolation=cv2.INTER_AREA
            )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise ValueError("JPEG encode failed")
        return buf.tobytes()

    def submit_jpeg(self, jpeg: bytes) -> None:
        """Hand one frame to the worker, replacing any frame it hasn't taken."""
        self._frame_ids += 1
        item = {"frame_id": self._frame_ids, "sent_at": time.time(), "jpeg": jpeg}
        try:
            self._frame_q.put_nowait(item)
        except queue.Full:
            self.stats["frames_dropped"] += 1
            _put_latest(self._frame_q, item)
        self.stats["frames_sent"] += 1

    def grab_frame(self) -> tuple[bytes, float] | None:
        """Newest raw camera JPEG and its age in seconds, for the VLM."""
        if self.latest_jpeg is None:
            return None
        return self.latest_jpeg, time.time() - self.latest_jpeg_at

    # ── Results out ─────────────────────────────────────────────────
    async def _poll_results(self) -> None:
        while self.state != "closed":
            try:
                msg = self._result_q.get_nowait()
            except queue.Empty:
                if self._proc is not None and not self._proc.is_alive():
                    await self._handle_crash()
                await asyncio.sleep(0.02)
                continue
            self._handle(msg)

    def _handle(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "ready":
            self.state = "ready"
            self.device = msg.get("device")
            self.stats["load_s"] = msg.get("load_s")
            logger.info(
                "vision ready in %ss on %s (depth=%s)",
                msg.get("load_s"),
                self.device,
                msg.get("depth"),
            )
        elif kind == "scene":
            now = time.time()
            scene = msg["scene"]
            self.latest_scene = scene
            self.stats["scenes"] += 1
            perf = scene.get("performance", {})
            self._recent.append(
                {
                    "t": now,
                    "inference_ms": perf.get("total_ms", 0.0),
                    "queue_ms": (msg["received_at"] - msg["sent_at"]) * 1000,
                    "end_to_end_ms": (now - msg["sent_at"]) * 1000,
                }
            )
            if self.on_scene:
                try:
                    self.on_scene(scene)
                except Exception:
                    logger.exception("on_scene callback failed")
            if self.on_overlay and msg.get("overlay"):
                try:
                    self.on_overlay(msg["overlay"])
                except Exception:
                    logger.exception("on_overlay callback failed")
        elif kind == "error":
            self.stats["errors"] += 1
            self.error = msg.get("error")
            logger.warning("vision worker error: %s", self.error)
            if self.state == "starting":
                self.state = "failed"  # model load failed; don't retry blindly

    async def _handle_crash(self) -> None:
        if self.state in ("failed", "closed"):
            return
        code = self._proc.exitcode if self._proc else None
        logger.warning("vision process exited (code %s)", code)
        self._proc = None
        if self._restarts >= self.max_restarts:
            self.state = "failed"
            self.error = f"vision process crashed {self._restarts + 1} times"
            return
        self._restarts += 1
        self.stats["restarts"] = self._restarts
        self.state = "idle"
        await asyncio.sleep(min(2**self._restarts, 10))
        if self.state == "idle":
            await self.start()

    # ── Read side ───────────────────────────────────────────────────
    def current_scene(self) -> dict[str, Any] | None:
        """Scene for the LLM, or None when there is no LiveKit camera."""
        if not self.has_camera:
            return None
        if self.state == "failed":
            return {"status": "error", "error": f"VisionGuide failed: {self.error}"}
        if self.state != "ready":
            return {
                "status": "error",
                "error": "The camera is connected but the vision models are still "
                "loading. Try again in a few seconds.",
            }
        if self.latest_scene is None:
            return {"status": "error", "error": "No camera frame processed yet."}
        age = time.time() - float(self.latest_scene.get("timestamp", 0))
        if age > self.scene_max_age_s:
            return {
                "status": "error",
                "error": f"The latest camera scene is {age:.0f}s old; the camera "
                "feed appears to have stopped.",
            }
        return {"status": "success", "age_s": round(age, 1), **self.latest_scene}

    def metrics(self) -> dict[str, Any]:
        recent = list(self._recent)
        out: dict[str, Any] = {**self.stats, "state": self.state, "device": self.device}
        if self.stats["frames_sent"]:
            out["stale_frame_rate"] = round(
                self.stats["frames_dropped"] / self.stats["frames_sent"], 3
            )
        if len(recent) >= 2:
            span = recent[-1]["t"] - recent[0]["t"]
            out["fps"] = round((len(recent) - 1) / span, 1) if span > 0 else None
        for key in ("inference_ms", "queue_ms", "end_to_end_ms"):
            if recent:
                out[f"avg_{key}"] = round(sum(r[key] for r in recent) / len(recent), 1)
        return out


class OverlayPublisher:
    """Streams the annotated vision frames back into the room as a video track.

    Anyone watching the room (a sighted helper, the Agents Playground, a judge)
    sees exactly what the agent sees: boxes with distances, free space per
    column, the direction arrow and the current guidance. Frames are decoded
    in a thread and dropped, never queued, when the previous one is still
    being published.
    """

    def __init__(self, room: Any, *, track_name: str = "visionguide") -> None:
        self.room = room
        self.track_name = track_name
        self.frames = 0
        self._source: Any = None
        self._track: Any = None
        self._publication: Any = None
        self._size: tuple[int, int] | None = None
        self._busy = False
        self._closed = False
        self._task: asyncio.Task | None = None

    def push(self, jpeg: bytes) -> None:
        if self._busy or self._closed:
            return
        self._busy = True
        self._task = asyncio.create_task(self._publish(jpeg), name="overlay_publish")

    async def _publish(self, jpeg: bytes) -> None:
        from livekit import rtc

        try:
            rgba, w, h = await asyncio.to_thread(self._decode, jpeg, self._size)
            if self._source is None:
                self._size = (w, h)
                self._source = rtc.VideoSource(w, h)
                self._track = rtc.LocalVideoTrack.create_video_track(
                    self.track_name, self._source
                )
                options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
                self._publication = await self.room.local_participant.publish_track(
                    self._track, options
                )
                logger.info("publishing VisionGuide overlay track %sx%s", w, h)
            self._source.capture_frame(
                rtc.VideoFrame(w, h, rtc.VideoBufferType.RGBA, rgba)
            )
            self.frames += 1
        except Exception:
            logger.exception("overlay publish failed")
            self._closed = True  # don't spam errors every frame
        finally:
            self._busy = False

    @staticmethod
    def _decode(jpeg: bytes, size: tuple[int, int] | None) -> tuple[bytes, int, int]:
        import cv2

        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode overlay frame")
        if size is not None and (bgr.shape[1], bgr.shape[0]) != size:
            bgr = cv2.resize(bgr, size)
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        return rgba.tobytes(), rgba.shape[1], rgba.shape[0]

    async def aclose(self) -> None:
        self._closed = True
        if self._publication is not None:
            with contextlib.suppress(Exception):
                await self.room.local_participant.unpublish_track(self._publication.sid)
        self._publication = None
