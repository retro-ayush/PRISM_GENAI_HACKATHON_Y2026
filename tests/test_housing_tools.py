"""Housing tool interface: optional filters, canonical filter names, and
normalisation of spoken values, checked at the validation boundary before
anything reaches the backend."""

import json

import pytest

from coordinator import Coordinator
from latency_injector import LatencyInjector
from mock_apis import register_mock_apis
from tool_registry import ToolRegistry, ToolValidationError


@pytest.fixture
def registry() -> ToolRegistry:
    return register_mock_apis(ToolRegistry(), LatencyInjector("instant"))


def validate(registry: ToolRegistry, tool: str, args: dict) -> dict:
    return registry.get(tool).validate(args)


# ── search_apartments: every filter optional, nothing invented ──────────
def test_search_requires_no_filter(registry) -> None:
    assert registry.get("search_apartments").parameters["required"] == []


@pytest.mark.parametrize(
    "args",
    [
        {"city": "Portland"},
        {"bedrooms": 2},
        {"max_price": 1800},
        {"city": "Portland", "bedrooms": 2, "max_price": 1800},
        {"city": "Miami", "max_price": 1500, "pets_allowed": True},
    ],
)
def test_search_keeps_exactly_the_given_filters(registry, args) -> None:
    assert validate(registry, "search_apartments", args) == args


def test_search_with_no_filters_adds_none(registry) -> None:
    assert validate(registry, "search_apartments", {}) == {}


@pytest.mark.parametrize(
    "raw", ["$1,800", "1,800", "1800", "1800 dollars", "₹1,800", "$1,800 per month"]
)
def test_spoken_budget_becomes_full_number(registry, raw) -> None:
    args = validate(registry, "search_apartments", {"max_price": raw})
    assert args == {"max_price": 1800}
    assert args["max_price"] != 800


def test_search_rejects_non_canonical_names(registry) -> None:
    with pytest.raises(ToolValidationError, match="min_bedrooms"):
        validate(registry, "search_apartments", {"city": "Austin", "min_bedrooms": 2})


@pytest.mark.parametrize(
    "args",
    [{"max_price": "cheap"}, {"max_price": 0}, {"bedrooms": -1}, {"city": "  "}],
)
def test_search_rejects_malformed_values(registry, args) -> None:
    with pytest.raises(ToolValidationError):
        validate(registry, "search_apartments", args)


# ── update_search_filter: closed set of filter names, typed values ──────
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"filter_name": "max_price", "value": "$1,500"}, 1500),
        ({"filter_name": "min_bedrooms", "value": "3"}, 3),
        ({"filter_name": "pets_allowed", "value": "true"}, True),
        ({"filter_name": "neighborhood", "value": "Northside"}, "Northside"),
    ],
)
def test_filter_values_are_typed(registry, args, expected) -> None:
    clean = validate(registry, "update_search_filter", args)
    assert clean == {"filter_name": args["filter_name"], "value": expected}


@pytest.mark.parametrize("name", ["preferred_neighborhood", "bedrooms", "budget"])
def test_filter_rejects_unknown_names(registry, name) -> None:
    with pytest.raises(ToolValidationError, match="filter_name must be one of"):
        validate(registry, "update_search_filter", {"filter_name": name, "value": 2})


def test_filter_names_are_an_enum_in_the_llm_schema(registry) -> None:
    schema = registry.get("update_search_filter").schema()
    assert schema["parameters"]["properties"]["filter_name"]["enum"] == [
        "neighborhood",
        "min_bedrooms",
        "max_price",
        "pets_allowed",
    ]


def test_filter_rejects_wrong_value_type(registry) -> None:
    with pytest.raises(ToolValidationError, match="max_price"):
        validate(
            registry,
            "update_search_filter",
            {"filter_name": "max_price", "value": True},
        )


# ── Through the coordinator: invalid calls never reach the backend ──────
async def test_invalid_call_returns_correctable_error(registry, tmp_path) -> None:
    tool_log = tmp_path / "tools.log"
    coord = Coordinator(registry, "room1", tool_log_path=tool_log)
    coord.on_user_turn("only show places that allow pets")

    bad = await coord.run_tool(
        "update_search_filter", {"filter_name": "preferred_pets", "value": True}
    )
    assert bad["status"] == "error"
    assert "pets_allowed" in bad["error"]  # tells the LLM the valid names
    assert not tool_log.exists()  # nothing was executed or logged

    good = await coord.run_tool(
        "update_search_filter", {"filter_name": "pets_allowed", "value": True}
    )
    coord.close()
    assert good["status"] == "success"
    logged = json.loads(tool_log.read_text())
    assert logged["call"]["args"] == {"filter_name": "pets_allowed", "value": True}


async def test_partial_search_runs_and_logs_only_given_filters(
    registry, tmp_path
) -> None:
    tool_log = tmp_path / "tools.log"
    coord = Coordinator(registry, "room1", tool_log_path=tool_log)
    coord.on_user_turn("Search apartments in Denver with one bedroom")
    result = await coord.run_tool(
        "search_apartments", {"city": "Denver", "bedrooms": 1}
    )
    coord.close()
    assert result["status"] == "success"
    assert json.loads(tool_log.read_text())["call"]["args"] == {
        "city": "Denver",
        "bedrooms": 1,
    }
