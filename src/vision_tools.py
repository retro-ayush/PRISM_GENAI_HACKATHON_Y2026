"""VisionGuide tools the LLM can call, beyond the reflex path.

Two tiers of perception:

- the *fast* tier (YOLO26 detection + depth, ~30 ms/frame in the vision
  process) drives obstacle warnings and ``get_current_scene``;
- the *slow* tier sends the latest camera frame to a vision-language model
  (``describe_view``) for anything the detector can't answer: a rich
  description, reading a sign or a label, colours, what's on the table.
  It runs as a separate LLM request so the conversation context never holds
  images, and the main agent stays fast.

Plus small deterministic tools: find an object by name, guide the user to a
target, change how talkative VisionGuide is, and switch the spoken language.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from livekit.agents import inference, llm

from scene import find_in_scene, resolve_object_query, spoken_location, target_update
from scene_memory import MODES, SceneMemory
from tool_registry import ToolRegistry, ToolSpec

logger = logging.getLogger("vision.tools")

VLM_PROMPT = (
    "You describe a camera image for a blind user who is holding the camera "
    "at chest height, pointing forward. Answer their question in two or three "
    "short spoken sentences. Describe only what is clearly visible; say when "
    "something is unclear or cut off. Read any visible text exactly. Use "
    "left, right, ahead and near or far rather than pixel positions. A list "
    "of objects a detector found, with rough distances, is included: use it "
    "to anchor distances, and do not contradict it unless the image clearly "
    "shows otherwise. Never claim the path is safe."
)
_READ_WORDS = re.compile(
    r"\b(read|text|sign|label|says?|written|writing|number|price|menu|screen|display|letter|word)",
    re.I,
)

# Languages supported by both Deepgram Nova-3 (STT) and Cartesia Sonic-3 (TTS).
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ja": "Japanese",
    "ru": "Russian",
}
_LANGUAGE_NAMES = {name.lower(): code for code, name in LANGUAGES.items()}
_LANGUAGE_NAMES.update(
    {
        "हिंदी": "hi",
        "hindi": "hi",
        "español": "es",
        "français": "fr",
        "deutsch": "de",
        "日本語": "ja",
        "русский": "ru",
    }
)

_OBJ = {"type": "object", "properties": {}, "required": []}


def _params(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


@dataclass
class VisionContext:
    """What the tools need at call time. ``session`` is set after start."""

    raw_scene: Callable[[], dict[str, Any]]
    memory: SceneMemory | None = None
    grab_frame: Callable[[], tuple[bytes, float] | None] | None = None
    session: Any = None
    vlm_model: str = "openai/gpt-4o-mini"
    vlm_timeout_s: float = 15.0
    scene_max_age_s: float = 5.0
    language: str = "en"
    _vlm: Any = field(default=None, repr=False)

    @property
    def vlm(self):
        if self._vlm is None:
            self._vlm = inference.LLM(model=self.vlm_model)
        return self._vlm


def register_vision_tools(registry: ToolRegistry, vc: VisionContext) -> ToolRegistry:
    registry.register(
        ToolSpec(
            name="describe_view",
            description="Look at the live camera image with a vision model and "
            "answer a question about it: a detailed description of the "
            "surroundings, reading a sign, label, screen or document, colours, "
            "what is on a table, whether a door is open. Slower (a few seconds) "
            "than get_current_scene; use it for anything the detector's object "
            "list cannot answer. Pass the user's question.",
            kind="read_only",
            parameters=_params(
                {"question": {"type": "string", "description": "What to look for"}},
                ["question"],
            ),
            handler=lambda question: describe_view(vc, question),
        )
    )
    registry.register(
        ToolSpec(
            name="find_object",
            description="Locate a named object in the camera view (phone, bag, "
            "chair, door, person, bottle...). Returns where it is relative to "
            "the user and roughly how far. Use when the user asks where "
            "something is or whether something is nearby.",
            kind="read_only",
            parameters=_params(
                {"name": {"type": "string", "description": "Object to look for"}},
                ["name"],
            ),
            handler=lambda name: find_object(vc, name),
        )
    )
    registry.register(
        ToolSpec(
            name="guide_to",
            description='Start guiding the user toward an object ("guide me to the '
            'chair", "take me to the door"). VisionGuide then repeats the '
            "object's direction every few seconds until the user reaches it or "
            "says stop. Returns the first direction to speak.",
            kind="read_only",
            parameters=_params(
                {"target": {"type": "string", "description": "Object to walk to"}},
                ["target"],
            ),
            handler=lambda target: guide_to(vc, target),
        )
    )
    registry.register(
        ToolSpec(
            name="stop_guidance",
            description="Stop guiding the user toward the current target.",
            kind="read_only",
            parameters=_OBJ,
            handler=lambda: stop_guidance(vc),
        )
    )
    registry.register(
        ToolSpec(
            name="set_guidance_mode",
            description="Change how talkative VisionGuide's automatic warnings "
            "are: 'quiet' (only urgent dangers), 'normal', 'verbose' (also "
            "objects passing close on either side), 'pause' (silence all "
            "automatic warnings) or 'resume'.",
            kind="read_only",
            parameters=_params(
                {
                    "mode": {
                        "type": "string",
                        "enum": [*MODES, "pause", "resume"],
                    }
                },
                ["mode"],
            ),
            handler=lambda mode: set_guidance_mode(vc, mode),
        )
    )
    registry.register(
        ToolSpec(
            name="set_language",
            description="Switch the language the assistant listens and speaks "
            "in, e.g. when the user asks to speak Hindi. Supported: English, "
            "Hindi, Spanish, French, German, Italian, Portuguese, Dutch, "
            "Japanese, Russian.",
            kind="read_only",
            parameters=_params(
                {
                    "language": {
                        "type": "string",
                        "description": "Language name or code",
                    }
                },
                ["language"],
            ),
            handler=lambda language: set_language(vc, language),
        )
    )
    return registry


# ── Handlers ─────────────────────────────────────────────────────────


_CAMERA_OFF_HINT = (
    "The camera is unavailable: tell the user and do not call other vision "
    "tools in this turn."
)


def _camera_error(vc: VisionContext) -> dict[str, Any] | None:
    scene = vc.raw_scene()
    if scene.get("status") != "success":
        return {**scene, "hint": _CAMERA_OFF_HINT}
    return None


def _scene_summary(scene: dict[str, Any]) -> str:
    parts = []
    for o in (scene.get("objects") or [])[:6] + (scene.get("others") or [])[:4]:
        if o.get("source") == "depth":
            continue
        parts.append(f"{o['label']} {spoken_location(o)}")
    return "; ".join(parts) if parts else "nothing detected"


async def describe_view(vc: VisionContext, question: str) -> dict[str, Any]:
    frame = vc.grab_frame() if vc.grab_frame else None
    if frame is None:
        return {
            "status": "error",
            "error": "No live camera frame is available, so I can't look at the view.",
            "hint": _CAMERA_OFF_HINT,
        }
    jpeg, age = frame
    if age > vc.scene_max_age_s:
        return {
            "status": "error",
            "error": f"The last camera frame is {age:.0f}s old; the camera seems stopped.",
        }
    scene = vc.raw_scene()
    summary = (
        _scene_summary(scene) if scene.get("status") == "success" else "unavailable"
    )
    detail = "high" if _READ_WORDS.search(question) else "low"
    ctx = llm.ChatContext()
    ctx.add_message(role="system", content=VLM_PROMPT)
    ctx.add_message(
        role="user",
        content=[
            f"Question: {question}\nDetector found: {summary}.",
            llm.ImageContent(
                image="data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
                inference_detail=detail,
            ),
        ],
    )
    try:
        text = await asyncio.wait_for(_vlm_answer(vc, ctx), timeout=vc.vlm_timeout_s)
    except TimeoutError:
        return {"status": "error", "error": "Looking at the image took too long."}
    except Exception as e:
        logger.exception("describe_view failed")
        return {"status": "error", "error": f"The vision model failed: {e!r}"}
    return {"status": "success", "description": text, "frame_age_s": round(age, 1)}


async def _vlm_answer(vc: VisionContext, ctx: llm.ChatContext) -> str:
    parts: list[str] = []
    async with vc.vlm.chat(chat_ctx=ctx) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                parts.append(chunk.delta.content)
    return "".join(parts).strip()


def find_object(vc: VisionContext, name: str) -> dict[str, Any]:
    clean, labels = resolve_object_query(name)
    if not labels:
        return {
            "status": "unknown_object",
            "name": clean or name,
            "hint": "The detector has no class for this; call describe_view with "
            f"the question 'Where is the {clean or name}?' instead.",
        }
    if (err := _camera_error(vc)) is not None:
        return err
    scene = vc.raw_scene()
    matches = find_in_scene(scene, labels)
    if not matches:
        return {
            "status": "not_found",
            "name": clean,
            "hint": f"No {clean} is in the camera view right now. Suggest turning "
            "slowly left or right and asking again.",
        }
    return {
        "status": "success",
        "name": clean,
        "count": len(matches),
        "matches": [
            {
                "label": o["label"],
                "location": spoken_location(o, elevation=True),
                "position": o["position"],
                "clock": o.get("clock"),
                "distance_m": o.get("distance_m"),
            }
            for o in matches[:3]
        ],
    }


def guide_to(vc: VisionContext, target: str) -> dict[str, Any]:
    clean, labels = resolve_object_query(target)
    if not labels:
        return {
            "status": "unknown_object",
            "name": clean or target,
            "hint": "The detector cannot track this object, so guidance to it is "
            "not possible. Offer describe_view instead.",
        }
    if vc.memory is None:
        return {"status": "error", "error": "VisionGuide is not running."}
    if (err := _camera_error(vc)) is not None:
        return err
    vc.memory.set_target(clean, labels)
    matches = find_in_scene(vc.raw_scene(), labels)
    if matches:
        first = target_update(clean, matches[0])
    else:
        first = f"I don't see a {clean} in view yet. Turn slowly so I can look for it."
    return {
        "status": "success",
        "target": clean,
        "visible": bool(matches),
        "say_now": first,
        "note": "Guidance updates will be spoken automatically every few seconds.",
    }


def stop_guidance(vc: VisionContext) -> dict[str, Any]:
    name = vc.memory.clear_target() if vc.memory else None
    if name is None:
        return {
            "status": "success",
            "note": "No guidance was active.",
            "say": "Guidance is off.",
        }
    return {
        "status": "success",
        "stopped": name,
        "say": f"Stopped guiding you to the {name}.",
    }


def set_guidance_mode(vc: VisionContext, mode: str) -> dict[str, Any]:
    if vc.memory is None:
        return {"status": "error", "error": "VisionGuide is not running."}
    mode = mode.lower().strip()
    if mode == "pause":
        vc.memory.paused = True
        return {
            "status": "success",
            "mode": "paused",
            "note": "Automatic warnings off.",
        }
    if mode == "resume":
        vc.memory.paused = False
        return {
            "status": "success",
            "mode": vc.memory.mode,
            "note": "Warnings back on.",
        }
    try:
        vc.memory.set_mode(mode)
    except ValueError as e:
        return {"status": "error", "error": str(e)}
    vc.memory.paused = False
    return {"status": "success", "mode": mode}


def set_language(vc: VisionContext, language: str) -> dict[str, Any]:
    key = language.strip().lower()
    code = key if key in LANGUAGES else _LANGUAGE_NAMES.get(key)
    if code is None:
        return {
            "status": "error",
            "error": f"Unsupported language {language!r}. Supported: "
            + ", ".join(LANGUAGES.values()),
        }
    session = vc.session
    if session is not None:
        try:
            if session.tts is not None:
                session.tts.update_options(language=code)
            if session.stt is not None and hasattr(session.stt, "update_options"):
                session.stt.update_options(language=code)
        except Exception as e:
            logger.exception("language switch failed")
            return {"status": "error", "error": f"Could not switch voice: {e!r}"}
    vc.language = code
    name = LANGUAGES[code]
    return {
        "status": "success",
        "language": name,
        "instruction": f"From now on, reply only in {name}, including this reply.",
    }
