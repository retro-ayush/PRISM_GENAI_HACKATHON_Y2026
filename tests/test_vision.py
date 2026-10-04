"""VisionGuide tests: scene building, scene memory, and process isolation.

The bridge tests spawn a real worker process (the torch-free fake worker), so
they exercise the actual IPC, latest-frame queue and lifecycle code.
"""

import asyncio
import sys
import time

import numpy as np

from scene import RawDetection, build_scene, describe_distance, object_distance
from scene_memory import SceneMemory
from session_metrics import InterruptionTracker
from vision_bridge import VisionBridge
from vision_worker import run_fake_worker

W, H = 640, 480


def box(position: str, size: int = 200) -> tuple[float, float, float, float]:
    cx = {"left": 100, "center": 320, "right": 540}[position]
    return (cx - size / 2, 140.0, cx + size / 2, 140.0 + size)


def depth_with(region: tuple, distance: float) -> np.ndarray:
    depth = np.full((H, W), 20.0, dtype=np.float32)
    x1, y1, x2, y2 = (int(v) for v in region)
    depth[y1:y2, x1:x2] = distance
    return depth


def obj(label: str, position: str, proximity: str, distance=None) -> dict:
    return {
        "label": label,
        "confidence": 0.9,
        "position": position,
        "distance_m": distance,
        "proximity": proximity,
    }


# ── Scene building ──────────────────────────────────────────────────


def test_scene_uses_depth_for_distance_and_proximity():
    b = box("center")
    scene = build_scene([RawDetection("chair", 0.9, b)], W, H, depth_with(b, 1.24))
    [chair] = scene["objects"]
    expected = obj("chair", "center", "close", 1.2) | {"pass_on": "left"}
    assert {k: chair[k] for k in expected} == expected
    assert chair["clock"] == "12 o'clock" and chair["elevation"] == "level"
    assert scene["depth_available"] is True
    assert scene["guidance"].startswith("Warning: a chair directly ahead")


def test_scene_falls_back_to_box_size_without_depth():
    scene = build_scene([RawDetection("person", 0.8, box("left", 400))], W, H)
    [person] = scene["objects"]
    assert person["distance_m"] is None
    assert person["proximity"] == "very close"
    assert scene["depth_available"] is False


def test_scene_filters_and_sorts_by_urgency():
    far_left = box("left", 60)
    near_center = box("center")
    depth = depth_with(far_left, 6.0)
    depth[140:340, 220:420] = 1.5
    scene = build_scene(
        [
            RawDetection("person", 0.9, far_left),
            RawDetection("chair", 0.9, near_center),
            RawDetection("person", 0.2, box("right")),  # low confidence
            RawDetection("toothbrush", 0.9, box("right")),  # not navigation-relevant
        ],
        W,
        H,
        depth,
    )
    assert [o["label"] for o in scene["objects"]] == ["chair", "person"]


def test_object_distance_ignores_box_edges_and_invalid_pixels():
    b = (100.0, 100.0, 300.0, 300.0)
    depth = np.full((H, W), 9.0, dtype=np.float32)
    depth[150:250, 150:250] = 2.0  # object core
    depth[150:160, 150:250] = np.nan
    assert object_distance(depth, b) == 2.0


def test_spoken_distance_is_approximate():
    assert describe_distance(obj("x", "center", "close", 1.2)) == "about 1 metre away"
    assert (
        describe_distance(obj("x", "center", "medium", 3.36)) == "about 3.5 metres away"
    )
    assert (
        describe_distance(obj("x", "center", "very close", 0.6))
        == "less than a metre away"
    )
    assert describe_distance(obj("x", "center", "far")) == "far"


def test_empty_scene_never_claims_path_is_safe():
    guidance = build_scene([], W, H)["guidance"].lower()
    assert "safe" not in guidance and "clear" not in guidance


# ── Scene memory ────────────────────────────────────────────────────


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def scene_of(*objects) -> dict:
    return {"objects": list(objects)}


def test_new_obstacle_announced_once_after_confirmation():
    memory = SceneMemory(confirm_frames=2, clock=Clock())
    chair = obj("chair", "center", "medium", 3.0)
    assert memory.update(scene_of(chair)) is None  # one frame could be a flicker
    assert "chair ahead" in memory.update(scene_of(chair)).text
    for _ in range(5):
        assert memory.update(scene_of(chair)) is None  # unchanged: stay silent


def test_approaching_obstacle_is_announced_again():
    clock = Clock()
    memory = SceneMemory(confirm_frames=1, min_repeat_s=4, clock=clock)
    assert memory.update(scene_of(obj("chair", "center", "medium", 3.0)))
    clock.t = 5
    message = memory.update(scene_of(obj("chair", "center", "close", 1.5)))
    assert "getting closer to the chair" in message.text
    # Moving back out is not news.
    clock.t = 10
    assert memory.update(scene_of(obj("chair", "center", "medium", 3.0))) is None


def test_very_close_overrides_repeat_cooldown():
    clock = Clock()
    memory = SceneMemory(confirm_frames=1, min_repeat_s=10, clock=clock)
    memory.update(scene_of(obj("person", "center", "close", 1.5)))
    clock.t = 1
    assert memory.update(scene_of(obj("person", "center", "medium", 2.5))) is None
    assert (
        "very close"
        in memory.update(scene_of(obj("person", "center", "very close", 0.6))).text
    )


def test_off_path_and_far_objects_are_not_announced():
    memory = SceneMemory(confirm_frames=1, clock=Clock())
    for _ in range(3):
        assert (
            memory.update(
                scene_of(
                    obj("person", "left", "close", 1.5),
                    obj("bus", "center", "far", 9.0),
                )
            )
            is None
        )


def test_forgotten_obstacle_is_announced_when_it_returns():
    memory = SceneMemory(confirm_frames=1, forget_after=2, clock=Clock())
    chair = obj("chair", "center", "close", 1.5)
    assert memory.update(scene_of(chair))
    memory.update(scene_of())
    memory.update(scene_of())
    assert memory.update(scene_of(chair))


def test_announcement_waits_until_quiet():
    memory = SceneMemory(confirm_frames=1, clock=Clock())
    chair = obj("chair", "center", "close", 1.5)
    assert memory.update(scene_of(chair), announce=False) is None  # user talking
    assert memory.update(scene_of(chair), announce=True)  # still pending, now said


def test_most_urgent_change_is_spoken_first():
    memory = SceneMemory(confirm_frames=1, clock=Clock())
    message = memory.update(
        scene_of(
            obj("chair", "center", "medium", 3.0),
            obj("person", "center", "very close", 0.7),
        )
    )
    assert "person" in message.text and message.urgent
    assert (
        "chair" in memory.update(scene_of(obj("chair", "center", "medium", 3.0))).text
    )


# ── Interruption metrics ────────────────────────────────────────────


def test_interruption_cancel_latency():
    events = []
    t = InterruptionTracker(lambda e, **d: events.append((e, d)))
    t.on_agent_state("speaking", 10.0)
    t.on_user_state("speaking", 11.0)
    t.on_agent_state("listening", 11.25)
    assert t.summary()["interruptions"] == 1
    assert events == [("INTERRUPTION", {"cancel_ms": 250.0})]


def test_missed_and_false_interruptions():
    t = InterruptionTracker(missed_after_s=1.0)
    t.on_agent_state("speaking", 0.0)
    t.on_user_state("speaking", 1.0)
    t.on_user_state("listening", 2.5)  # talked over the agent; it never stopped
    t.on_false_interruption(resumed=True)
    s = t.summary()
    assert (
        s["interruptions"],
        s["missed_interruptions"],
        s["false_interruptions"],
    ) == (0, 1, 1)


def test_user_speech_without_agent_speech_is_not_an_interruption():
    t = InterruptionTracker()
    t.on_user_state("speaking", 0.0)
    t.on_user_state("listening", 2.0)
    t.on_agent_state("thinking", 2.1)
    t.on_agent_state("speaking", 2.5)
    t.on_agent_state("listening", 4.0)
    assert t.summary()["interruptions"] == 0


# ── Bridge / process isolation ──────────────────────────────────────


async def wait_for(predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.02)


async def test_vision_process_end_to_end():
    scenes = []
    bridge = VisionBridge(target=run_fake_worker, on_scene=scenes.append)
    bridge.has_camera = True  # stand-in for an attached LiveKit track
    try:
        assert bridge.current_scene()["status"] == "error"  # not started yet
        await bridge.start()
        await wait_for(lambda: bridge.state == "ready")
        assert bridge._proc.pid != __import__("os").getpid()

        for _ in range(3):
            seen = len(scenes)
            bridge.submit_jpeg(b"frame")
            await wait_for(lambda seen=seen: len(scenes) > seen)

        result = bridge.current_scene()
        assert result["status"] == "success"
        assert result["objects"][0]["label"] == "chair"
        assert result["objects"][0]["distance_m"] == 3.0  # 4.5 - 0.5 * 3
        metrics = bridge.metrics()
        assert metrics["scenes"] == 3 and metrics["frames_sent"] == 3
        assert "avg_end_to_end_ms" in metrics
        # The agent process never loads the heavy vision stack.
        assert "torch" not in sys.modules and "ultralytics" not in sys.modules
    finally:
        await bridge.aclose()
    assert bridge._proc is None


async def test_event_loop_stays_responsive_while_vision_loads():
    bridge = VisionBridge(target=run_fake_worker, config={"fake_load_s": 1.0})
    worst_lag = 0.0
    try:
        await bridge.start()
        while bridge.state != "ready":
            t = time.perf_counter()
            await asyncio.sleep(0.01)
            worst_lag = max(worst_lag, time.perf_counter() - t - 0.01)
    finally:
        await bridge.aclose()
    assert worst_lag < 0.1


async def test_latest_frame_wins_when_worker_is_busy():
    bridge = VisionBridge(target=run_fake_worker, config={"fake_load_s": 0.5})
    try:
        await bridge.start()
        # Worker is still loading: frames pile up on a size-1 queue.
        for _ in range(5):
            bridge.submit_jpeg(b"frame")
        assert bridge.stats["frames_dropped"] >= 3
        scenes = []
        bridge.on_scene = scenes.append
        await wait_for(lambda: scenes)
        assert scenes[0]["frame_id"] == 5  # only the newest frame was processed
    finally:
        await bridge.aclose()


async def test_stale_scene_is_reported():
    bridge = VisionBridge(scene_max_age_s=1.0)
    bridge.has_camera = True
    bridge.state = "ready"
    bridge.latest_scene = {"timestamp": time.time() - 10, "objects": []}
    assert "old" in bridge.current_scene()["error"]
    bridge.has_camera = False
    assert bridge.current_scene() is None  # falls back to scene.json


# ── Free space, walls and direction ─────────────────────────────────


def floor_depth() -> np.ndarray:
    """Open room: distant background, floor getting nearer toward the bottom."""
    depth = np.full((H, W), 6.0, dtype=np.float32)
    rows = np.linspace(6.0, 1.2, H // 2, dtype=np.float32)
    depth[H // 2 :] = rows[:, None]
    return depth


def test_floor_is_not_an_obstacle():
    scene = build_scene([], W, H, floor_depth())
    assert scene["path"]["recommendation"] == "straight"
    assert scene["path"]["wall_ahead"] is False
    assert scene["objects"] == []
    assert scene["guidance"] == "No obstacles detected ahead."


def test_wall_ahead_is_detected_with_a_way_around():
    depth = floor_depth()
    depth[:, W // 3 : 2 * W // 3] = 1.1  # flat surface filling the center
    scene = build_scene([], W, H, depth)
    path = scene["path"]
    assert path["wall_ahead"] is True
    assert path["clearance_m"]["center"] == 1.1
    assert path["recommendation"] in ("step_left", "step_right")
    [wall] = scene["objects"]
    assert wall["label"] == "wall" and wall["source"] == "depth"
    assert scene["guidance"].startswith("Warning: a wall directly ahead")
    assert "Move to your" not in scene["guidance"]  # uses the pass_on hint
    assert "Go to your" in scene["guidance"]


def test_wall_across_whole_view_means_stop():
    depth = np.full((H, W), 0.9, dtype=np.float32)
    scene = build_scene([], W, H, depth)
    assert scene["path"]["recommendation"] == "stop"
    assert "Stop" in scene["guidance"]


def test_person_ahead_is_passed_on_the_freer_side():
    b = box("center", 160)
    depth = floor_depth()
    depth[140:300, 240:400] = 2.5  # person in the middle, not blocking yet
    depth[:, 2 * W // 3 :] = 1.0  # right side cluttered and close
    scene = build_scene([RawDetection("person", 0.9, b)], W, H, depth)
    person = next(o for o in scene["objects"] if o["label"] == "person")
    assert person["pass_on"] == "left"
    assert "Go to your left" in scene["guidance"]


def test_detected_object_is_not_double_reported_as_obstacle():
    b = box("center")
    depth = floor_depth()
    depth[140:340, 220:420] = 1.0
    depth[:, W // 3 : 2 * W // 3][int(H * 0.25) : int(H * 0.7)] = 1.0
    scene = build_scene([RawDetection("chair", 0.9, b)], W, H, depth)
    assert [o["label"] for o in scene["objects"] if o["position"] == "center"] == [
        "chair"
    ]


def test_direction_without_depth_uses_detections():
    scene = build_scene(
        [
            RawDetection("person", 0.9, box("center", 400)),
            RawDetection("chair", 0.9, box("left", 300)),
        ],
        W,
        H,
    )
    assert scene["path"]["recommendation"] == "step_right"


def test_stop_is_announced_once_and_urgently():
    memory = SceneMemory(confirm_frames=1, clock=Clock())
    blocked = {"objects": [], "path": {"recommendation": "stop"}}
    message = memory.update(blocked)
    assert message.urgent and message.text.startswith("Stop")
    assert memory.update(blocked) is None


def test_wall_announcement_includes_direction():
    memory = SceneMemory(confirm_frames=1, clock=Clock())
    wall = obj("wall", "center", "close", 1.2) | {"source": "depth", "pass_on": "right"}
    message = memory.update(
        {"objects": [wall], "path": {"recommendation": "step_right"}}
    )
    assert message.text == (
        "Heads up, a wall ahead, about 1 metre away. Go to your right."
    )
    assert not message.urgent


def test_busy_agent_only_hears_urgent_warnings():
    memory = SceneMemory(confirm_frames=1, clock=Clock())
    chair = obj("chair", "center", "medium", 3.0)
    assert memory.update(scene_of(chair), urgent_only=True) is None  # stays pending
    assert memory.update(scene_of(chair)).text.startswith("Heads up")
    near = obj("person", "center", "very close", 0.6)
    assert memory.update(scene_of(near), urgent_only=True).urgent


def test_wall_and_obstacle_flicker_is_one_announcement():
    memory = SceneMemory(confirm_frames=1, clock=Clock())
    a = obj("wall", "center", "close", 1.2) | {"source": "depth"}
    b = obj("obstacle", "center", "close", 1.3) | {"source": "depth"}
    assert memory.update(scene_of(a))
    assert memory.update(scene_of(b)) is None


def test_pass_side_avoids_other_people():
    depth = floor_depth()
    depth[100:400, 200:440] = 1.8  # person ahead
    depth[100:400, 470:620] = 1.8  # another person on the right
    scene = build_scene(
        [
            RawDetection(
                "person", 0.9, (200.0, 100.0, 440.0, 400.0)
            ),  # right of center
            RawDetection("person", 0.9, (470.0, 100.0, 620.0, 400.0)),
        ],
        W,
        H,
        depth,
    )
    ahead = next(o for o in scene["objects"] if o["position"] == "center")
    assert ahead["pass_on"] == "left"
