"""VisionGuide standalone mode: local webcam -> YOLO26 detection + depth -> scene.json.

The main path is the voice agent reading the user's LiveKit camera in its own
vision process (src/vision_bridge.py). This script is the offline/demo
fallback: it runs the same VisionEngine on a local webcam in a separate
process and writes vision_guide/scene.json, which the agent's
get_current_scene tool reads when no LiveKit camera is connected.

    uv run python vision_guide/vision.py [--camera 0] [--no-window]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vision_worker import VisionEngine, draw_overlay  # noqa: E402

SCENE_FILE = Path(__file__).resolve().parent / "scene.json"


def write_scene(scene: dict) -> None:
    # Atomic replace, so the agent never reads a half-written file.
    tmp = SCENE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    os.replace(tmp, SCENE_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--interval", type=float, default=0.5, help="scene.json write period (s)"
    )
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--device", default=os.getenv("VISION_DEVICE", "auto"))
    args = parser.parse_args()

    print("Loading YOLO26 detection + depth models...")
    t0 = time.perf_counter()
    engine = VisionEngine(
        detect_model=os.getenv("VISION_DETECT_MODEL", "yolo26n.pt"),
        depth_model=os.getenv("VISION_DEPTH_MODEL", "yolo26n-depth.pt") or None,
        device=args.device,
    )
    print(f"Models ready in {time.perf_counter() - t0:.1f}s on {engine.device}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")
    print(f"VisionGuide active. Scene -> {SCENE_FILE}. Press Q to quit.")

    last_write = 0.0
    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read camera frame")
                break
            frame_id += 1
            scene = engine.process(frame, frame_id=frame_id)

            now = time.time()
            if now - last_write >= args.interval:
                write_scene(scene)
                last_write = now
                objs = ", ".join(
                    f"{o['label']} {o['position']} {o['distance_m'] or o['proximity']}"
                    for o in scene["objects"]
                )
                print(
                    f"[{scene['performance']['total_ms']:5.1f} ms] {scene['guidance']}  ({objs})"
                )

            if not args.no_window:
                cv2.imshow("VisionGuide", draw_overlay(frame, scene))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("VisionGuide stopped.")


if __name__ == "__main__":
    main()
