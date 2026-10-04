"""Schema-driven tool registry.

Every tool is described by a JSON schema plus a ``kind``:
- ``read_only``: safe to repeat; results are still version-checked.
- ``state_modifying``: committed at most once per idempotency key, and only if
  the user's intent hasn't changed while the call was in flight.

Tools can be registered in code or loaded from a JSON file at runtime, so the
agent doesn't assume a fixed tool set.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ToolKind = Literal["read_only", "state_modifying"]

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


class ToolValidationError(ValueError):
    pass


@dataclass
class ToolSpec:
    name: str
    description: str
    kind: ToolKind
    parameters: dict[str, Any]
    # Executes the tool. Sync or async; receives the validated arguments.
    handler: Callable[..., dict[str, Any] | Awaitable[dict[str, Any]]]
    # Optional simulated backend delay, awaited before the handler runs. For
    # mutations, the coordinator re-checks intent after this and before commit.
    latency: Callable[[str], Awaitable[Any]] | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def mutating(self) -> bool:
        return self.kind == "state_modifying"

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Check args against the schema; drop unknown keys and fill defaults."""
        props: dict[str, Any] = self.parameters.get("properties", {})
        required: list[str] = self.parameters.get("required", [])
        missing = [p for p in required if args.get(p) is None]
        if missing:
            raise ToolValidationError(f"missing required argument(s): {missing}")

        clean: dict[str, Any] = {}
        for key, prop in props.items():
            if args.get(key) is None:
                if "default" in prop:
                    clean[key] = prop["default"]
                continue
            value = args[key]
            types = prop.get("type")
            types = [types] if isinstance(types, str) else (types or [])
            allowed = tuple(t for name in types for t in _JSON_TYPES.get(name, ()))
            # bool is a subclass of int; don't accept True as a number.
            if allowed and (
                not isinstance(value, allowed)
                or (isinstance(value, bool) and bool not in allowed)
            ):
                raise ToolValidationError(
                    f"argument {key!r} must be of type {types}, got {value!r}"
                )
            if "enum" in prop and value not in prop["enum"]:
                raise ToolValidationError(
                    f"argument {key!r} must be one of {prop['enum']}"
                )
            clean[key] = value
        return clean

    async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        result = self.handler(**args)
        if inspect.isawaitable(result):
            result = await result
        return result


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.kind not in ("read_only", "state_modifying"):
            raise ValueError(f"tool {spec.name!r} has invalid kind {spec.kind!r}")
        if spec.parameters.get("type", "object") != "object":
            raise ValueError(f"tool {spec.name!r} parameters must be an object schema")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def load_json(self, path: str | Path) -> list[str]:
        """Register tools from a JSON file (a list of tool schemas).

        Each entry: ``{"name", "description", "kind", "parameters",
        "mock_response"?}``. Without a live backend, a loaded tool returns its
        ``mock_response`` merged with the arguments it was called with.
        """
        entries = json.loads(Path(path).read_text(encoding="utf-8"))
        names = []
        for entry in entries:
            mock = entry.get("mock_response", {"status": "success"})

            def handler(_mock: dict = mock, **kwargs: Any) -> dict[str, Any]:
                return {**_mock, "request": kwargs}

            self.register(
                ToolSpec(
                    name=entry["name"],
                    description=entry["description"],
                    kind=entry.get("kind", "read_only"),
                    parameters=entry.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                    handler=handler,
                )
            )
            names.append(entry["name"])
        return names
