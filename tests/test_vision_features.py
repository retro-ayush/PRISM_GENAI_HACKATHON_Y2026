"""Tests for VisionGuide's navigation features: drop-offs, clock bearings,
object search, guidance modes, fast-approach warnings, guide-to-target,
the overlay, the alert tone and the session report."""

import asyncio
import importlib.util
import sys
import time
import wave
from pathlib import Path

import numpy as np

from alert_tone import ensure_alert_wav
from scene import (
    RawDetection,
    build_scene,
    clock_position,
    detect_drop_off,
    find_in_scene,
    resolve_object_query,
    spoken_location,
    target_update,
)
from scene_memory import SceneMemory
from vision_bridge import VisionBridge
from vision_tools import (
    VisionContext,
    describe_view,
    find_object,
    guide_to,
    set_guidance_mode,
    set_language,
    stop_guidance,
)
from vision_worker import draw_overlay, encode_overlay, run_fake_worker

W, H = 640, 480


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def obj(label, position, proximity, distance=None, **extra) -> dict:
    return {
        "label": label,
        "confidence": 0.9,
        "position": position,
        "clock": "12 o'clock",
        "elevation": "level",
        "distance_m": distance,
        "proximity": proximity,
        **extra,
    }


def scene_of(*objects, path=None, others=()) -> dict:
    return {
        "objects": list(objects),
        "others": list(others),
        "path": path or {"recommendation": "straight"},
    }


def rows(spec: list[tuple[int, int, float, float]]) -> np.ndarray:
    """Depth map from (row_from, row_to, depth_at_top, depth_at_bottom) bands."""
    depth = np.full((H, W), 8.0, dtype=np.float32)
    for r0, r1, top, bottom in spec:
        depth[r0:r1] = np.linspace(top, bottom, r1 - r0, dtype=np.float32)[:, None]
    return depth


# ── Geometry ─────────────────────────────────────────────────────────


def test_clock_positions():
    assert clock_position(0) == "12 o'clock"
    assert clock_position(-33) == "11 o'clock"
    assert clock_position(33) == "1 o'clock"
    assert clock_position(-90) == "9 o'clock"
    scene = build_scene([RawDetection("chair", 0.9, (0.0, 0.0, 60.0, 60.0))], W, H)
    [chair] = scene["objects"]
    assert chair["clock"] == "11 o'clock" and chair["elevation"] == "high"


def test_others_keeps_non_navigation_detections():
    scene = build_scene(
        [RawDetection("laptop", 0.9, (300.0, 380.0, 400.0, 470.0))], W, H
    )
    assert scene["objects"] == []
    [laptop] = scene["others"]
    assert laptop["label"] == "laptop" and laptop["elevation"] == "low"
    assert scene["guidance"] == "No obstacles detected ahead."


# ── Drop-offs ────────────────────────────────────────────────────────


def test_drop_off_detected_where_receding_floor_jumps_away():
    depth = rows([(360, 480, 1.6, 1.0), (240, 360, 5.0, 5.0)])
    assert detect_drop_off(depth[::4, ::4]) == 1.5
    scene = build_scene([], W, H, depth)
    assert scene["path"]["drop_off"] == {"distance_m": 1.5}
    assert scene["path"]["recommendation"] == "stop_drop"
    assert scene["guidance"].startswith("Careful: the ground may drop away")
    assert "cane" in scene["guidance"]


def test_obstacle_top_edge_is_not_a_drop_off():
    # Floor, then a flat vertical face (a box), then the far background.
    depth = rows([(440, 480, 1.2, 1.0), (300, 440, 1.3, 1.3), (240, 300, 5.0, 5.0)])
    assert detect_drop_off(depth[::4, ::4]) is None


def test_smooth_floor_has_no_drop_off():
    depth = rows([(240, 480, 6.0, 1.0)])
    assert detect_drop_off(depth[::4, ::4]) is None
    assert build_scene([], W, H, depth)["path"]["drop_off"] is None


# ── Finding things ───────────────────────────────────────────────────


def test_resolve_object_query_aliases_and_plurals():
    assert resolve_object_query("my phone") == ("phone", {"cell phone"})
    assert resolve_object_query("the chairs") == ("chairs", {"chair"})
    assert resolve_object_query("a red chair") == ("red chair", {"chair"})
    assert resolve_object_query("bags")[1] == {"backpack", "handbag", "suitcase"}
    assert resolve_object_query("the door") == ("door", set())


def test_find_in_scene_nearest_first_and_skips_depth_obstacles():
    scene = scene_of(
        obj("chair", "left", "medium", 3.0),
        obj("wall", "center", "close", 1.2, source="depth"),
        others=[obj("cell phone", "right", "close", 1.1, elevation="low")],
    )
    found = find_in_scene(scene, {"chair", "cell phone", "wall"})
    assert [o["label"] for o in found] == ["cell phone", "chair"]
    assert spoken_location(found[0], elevation=True) == (
        "on your right, about 1 metre away, low down"
    )


def test_target_update_phrasing():
    assert target_update("chair", obj("chair", "left", "medium", 3.0)) == (
        "The chair is on your left, about 3 metres away. Turn a little to your left."
    )
    assert target_update("chair", obj("chair", "center", "close", 1.4)) == (
        "The chair is just ahead, about 1.5 metres away. Keep going slowly."
    )


# ── Scene memory: hazards and modes ──────────────────────────────────


def test_drop_off_is_announced_once_and_urgently():
    memory = SceneMemory(confirm_frames=1, clock=Clock())
    path = {"recommendation": "stop_drop", "drop_off": {"distance_m": 1.5}}
    message = memory.update(scene_of(path=path))
    assert message.kind == "drop_off" and message.urgent
    assert "drop away" in message.text
    assert memory.update(scene_of(path=path)) is None


def test_fast_approaching_object_is_called_out():
    clock = Clock()
    memory = SceneMemory(confirm_frames=1, clock=clock)
    for t, d in ((0.0, 6.0), (0.5, 4.5), (1.0, 3.0)):
        clock.t = t
        message = memory.update(scene_of(obj("bicycle", "left", "medium", d)))
    assert message.kind == "approaching" and message.urgent
    assert (
        message.text == "Careful, a bicycle is coming toward you fast from your left."
    )
    clock.t = 1.5
    assert memory.update(scene_of(obj("bicycle", "left", "close", 1.6))) is None


def test_walking_toward_a_person_is_not_an_approach_warning():
    clock = Clock()
    memory = SceneMemory(confirm_frames=1, clock=clock)
    messages = []
    for t, d in ((0.0, 4.4), (0.5, 3.8), (1.0, 3.2)):  # ~1.2 m/s, walking pace
        clock.t = t
        messages.append(memory.update(scene_of(obj("person", "center", "medium", d))))
    assert all(m is None or m.kind != "approaching" for m in messages)


def test_quiet_mode_only_speaks_urgent_warnings():
    memory = SceneMemory(confirm_frames=1, mode="quiet", clock=Clock())
    assert memory.update(scene_of(obj("chair", "center", "medium", 3.0))) is None
    message = memory.update(scene_of(obj("person", "center", "very close", 0.6)))
    assert message is not None and message.urgent


def test_paused_memory_says_nothing_but_keeps_tracking():
    memory = SceneMemory(confirm_frames=1, clock=Clock())
    memory.paused = True
    near = obj("person", "center", "very close", 0.6)
    assert memory.update(scene_of(near)) is None
    memory.paused = False
    assert memory.update(scene_of(near)).urgent


def test_verbose_mode_mentions_objects_passing_close_on_a_side():
    side = obj("person", "left", "close", 1.3)
    normal = SceneMemory(confirm_frames=1, clock=Clock())
    assert normal.update(scene_of(side)) is None
    verbose = SceneMemory(confirm_frames=1, mode="verbose", clock=Clock())
    message = verbose.update(scene_of(side))
    assert message.kind == "side" and not message.urgent
    assert message.text == "A person on your left, about 1.5 metres away."
    assert verbose.update(scene_of(side)) is None


def test_path_open_is_mentioned_once_after_a_stop():
    memory = SceneMemory(confirm_frames=1, clock=Clock())
    blocked = scene_of(path={"recommendation": "stop"})
    assert memory.update(blocked).kind == "stop"
    assert memory.update(scene_of()) is None  # one open frame could be noise
    opened = memory.update(scene_of())
    assert opened.kind == "open" and opened.text == "The way ahead looks open now."
    assert memory.update(scene_of()) is None


# ── Scene memory: guide to target ────────────────────────────────────


def test_guide_to_target_repeats_direction_until_reached():
    clock = Clock()
    memory = SceneMemory(confirm_frames=1, clock=clock)
    memory.set_target("chair", {"chair"})
    chair_left = scene_of(others=[obj("chair", "left", "medium", 3.0)])
    first = memory.update(chair_left)
    assert first.kind == "target" and "Turn a little to your left" in first.text
    clock.t = 1.0
    assert memory.update(chair_left) is None  # same direction, not due yet
    clock.t = 3.2
    assert memory.update(chair_left).kind == "target"  # periodic repeat
    clock.t = 4.5
    ahead = memory.update(scene_of(others=[obj("chair", "center", "close", 1.4)]))
    assert ahead is not None and "just ahead" in ahead.text  # direction changed
    clock.t = 5.0
    reached = memory.update(
        scene_of(others=[obj("chair", "center", "very close", 0.6)])
    )
    assert reached.kind == "target_reached" and "reached the chair" in reached.text
    assert memory.target is None


def test_lost_target_asks_user_to_turn_once():
    clock = Clock()
    memory = SceneMemory(confirm_frames=1, forget_after=2, clock=clock)
    memory.set_target("chair", {"chair"})
    assert memory.update(scene_of()) is None
    lost = memory.update(scene_of())
    assert lost.kind == "target_lost" and "don't see a chair" in lost.text
    clock.t = 3.0
    assert memory.update(scene_of()) is None  # not nagging
    memory.clear_target()
    clock.t = 10.0
    assert memory.update(scene_of()) is None


def test_target_guidance_survives_quiet_mode():
    memory = SceneMemory(confirm_frames=1, mode="quiet", clock=Clock())
    memory.set_target("phone", {"cell phone"})
    message = memory.update(scene_of(others=[obj("cell phone", "right", "close", 1.2)]))
    assert message.kind == "target"


# ── Tools (no network) ───────────────────────────────────────────────


def _scene_ok() -> dict:
    return {
        "status": "success",
        **scene_of(
            obj("chair", "center", "medium", 3.0, pass_on="left"),
            others=[obj("cell phone", "right", "close", 1.1, elevation="low")],
        ),
    }


def test_find_object_tool():
    vc = VisionContext(raw_scene=_scene_ok)
    found = find_object(vc, "my phone")
    assert found["status"] == "success" and found["count"] == 1
    assert (
        found["matches"][0]["location"] == "on your right, about 1 metre away, low down"
    )
    assert find_object(vc, "the dog")["status"] == "not_found"
    unknown = find_object(vc, "door")
    assert unknown["status"] == "unknown_object" and "describe_view" in unknown["hint"]
    off = VisionContext(raw_scene=lambda: {"status": "error", "error": "no camera"})
    off_result = find_object(off, "phone")
    assert (
        off_result["status"] == "error"
        and "do not call other vision" in off_result["hint"]
    )


def test_guide_and_stop_tools():
    memory = SceneMemory(confirm_frames=1, clock=Clock())
    vc = VisionContext(raw_scene=_scene_ok, memory=memory)
    started = guide_to(vc, "the chair")
    assert started["status"] == "success" and started["visible"]
    assert started["say_now"].startswith("The chair is straight ahead")
    assert memory.target == "chair"
    stopped = stop_guidance(vc)
    assert stopped["stopped"] == "chair" and "Stopped guiding" in stopped["say"]
    assert memory.target is None
    assert guide_to(vc, "door")["status"] == "unknown_object"


def test_guidance_mode_tool():
    memory = SceneMemory(confirm_frames=1, clock=Clock())
    vc = VisionContext(raw_scene=_scene_ok, memory=memory)
    assert set_guidance_mode(vc, "pause")["mode"] == "paused" and memory.paused
    assert set_guidance_mode(vc, "resume")["mode"] == "normal" and not memory.paused
    assert set_guidance_mode(vc, "Quiet")["mode"] == "quiet" and memory.mode == "quiet"
    assert set_guidance_mode(vc, "loud")["status"] == "error"


class _Recorder:
    def __init__(self) -> None:
        self.calls = []

    def update_options(self, **kw) -> None:
        self.calls.append(kw)


class _FakeSession:
    def __init__(self) -> None:
        self.tts = _Recorder()
        self.stt = _Recorder()


def test_language_tool_switches_tts_and_stt():
    session = _FakeSession()
    vc = VisionContext(raw_scene=_scene_ok, session=session)
    result = set_language(vc, "Hindi")
    assert result["status"] == "success" and result["language"] == "Hindi"
    assert "reply only in Hindi" in result["instruction"]
    assert session.tts.calls == [{"language": "hi"}]
    assert session.stt.calls == [{"language": "hi"}]
    assert vc.language == "hi"
    assert set_language(vc, "Klingon")["status"] == "error"


async def test_describe_view_needs_a_live_frame():
    vc = VisionContext(raw_scene=_scene_ok)  # no grab_frame
    result = await describe_view(vc, "what is ahead?")
    assert result["status"] == "error" and "camera" in result["error"]
    stale = VisionContext(raw_scene=_scene_ok, grab_frame=lambda: (b"x", 30.0))
    result = await describe_view(stale, "what is ahead?")
    assert result["status"] == "error" and "old" in result["error"]


# ── Overlay, tone, report ────────────────────────────────────────────


def test_overlay_draws_on_the_frame():
    depth = np.full((H, W), 6.0, dtype=np.float32)
    depth[:, W // 3 : 2 * W // 3] = 1.1  # wall ahead
    scene = build_scene(
        [RawDetection("person", 0.9, (40.0, 100.0, 200.0, 400.0))], W, H, depth
    )
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    out = draw_overlay(frame, scene)
    assert out.shape == frame.shape and out.any()
    jpeg = encode_overlay(frame, scene)
    assert jpeg[:2] == b"\xff\xd8" and len(jpeg) > 1000


async def test_fake_worker_streams_overlay_frames():
    overlays = []
    bridge = VisionBridge(
        target=run_fake_worker, config={"overlay": True}, on_overlay=overlays.append
    )
    try:
        await bridge.start()
        deadline = time.monotonic() + 20
        while bridge.state != "ready":
            assert time.monotonic() < deadline
            await asyncio.sleep(0.02)
        bridge.submit_jpeg(b"frame")
        while not overlays:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.02)
        assert overlays[0][:2] == b"\xff\xd8"
        assert bridge.grab_frame() is None  # no camera frame was ever read
        assert "torch" not in sys.modules
    finally:
        await bridge.aclose()


def test_alert_tone_is_a_short_wav(tmp_path: Path):
    path = ensure_alert_wav(tmp_path / "alert.wav")
    with wave.open(path, "rb") as wav:
        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
        duration = wav.getnframes() / wav.getframerate()
    assert 0.2 < duration < 0.4
    assert ensure_alert_wav(tmp_path / "alert.wav") == path  # cached


def test_report_summarizes_a_trace():
    spec = importlib.util.spec_from_file_location(
        "report", Path(__file__).resolve().parents[1] / "scripts" / "report.py"
    )
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    t = 1000.0
    events = [
        {"ts": t, "session_id": "s1", "type": "USER_MESSAGE", "state_version": 1, "text": "hi", "decision": "new_request"},
        {"ts": t + 0.1, "session_id": "s1", "type": "TOOL_STARTED", "state_version": 1, "tool": "search_flights", "args": {}},
        {"ts": t + 0.2, "session_id": "s1", "type": "RESULT_ACCEPTED", "state_version": 1, "tool": "search_flights"},
        {"ts": t + 0.5, "session_id": "s1", "type": "RESPONSE_LATENCY", "state_version": 1, "latency_ms": 420.0},
        {"ts": t + 2.0, "session_id": "s1", "type": "INTERRUPTION", "state_version": 1, "cancel_ms": 180.0},
        {"ts": t + 3.0, "session_id": "s1", "type": "VISION_ANNOUNCEMENT", "state_version": 1, "text": "Heads up, a chair ahead.", "kind": "obstacle", "urgent": False},
        {"ts": t + 4.0, "session_id": "s1", "type": "SESSION_METRICS", "state_version": 1, "vision": {"device": "mps", "fps": 7.5, "avg_inference_ms": 31.0}},
    ]  # fmt: skip
    text = report.render(report.summarize(events), timeline=True)
    assert "Session `s1`" in text
    assert "p50 420 ms" in text and "Barge-ins: 1 handled" in text
    assert "| search_flights | 1 | 1 |" in text
    assert "Heads up, a chair ahead." in text and "Device: mps" in text
    assert "barge-in, stopped in 180 ms" in text
