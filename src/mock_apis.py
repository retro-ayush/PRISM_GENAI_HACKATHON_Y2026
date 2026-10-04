"""Simulated Travel, Finance, Housing and E-commerce APIs.

The handlers are pure functions: calling one is the "commit". Simulated network
latency is applied separately (see ``LatencyInjector``) so the coordinator can
re-check the user's intent between the delay and the commit of a mutation.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

from latency_injector import LatencyInjector
from tool_registry import ToolRegistry, ToolSpec, ToolValidationError

_booking_ids = itertools.count(1)


# ── Travel & Identity ───────────────────────────────────────────
def search_flights(destination: str, date: str, origin: str | None = None) -> dict:
    return {
        "status": "success",
        "flights": [
            {
                "flight_id": "FL123",
                "origin": origin,
                "destination": destination,
                "date": date,
                "price": 450.0,
            }
        ],
    }


def book_flight(passenger_name: str, flight_id: str = "FL123") -> dict:
    return {
        "status": "success",
        "booking_ref": f"B{next(_booking_ids):04d}",
        "flight_id": flight_id,
        "passenger": passenger_name,
    }


def update_identity_doc(doc_type: str, doc_number: str) -> dict:
    return {
        "status": "success",
        "updated_doc": doc_type,
        "masked_number": doc_number[-4:],
    }


# ── Finance & Billing ───────────────────────────────────────────
def get_card_benefits(card_type: str) -> dict:
    return {
        "status": "success",
        "card_type": card_type,
        "benefits": ["2% Cashback", "No Foreign Transaction Fee"],
    }


def get_exchange_rate(amount: float, from_currency: str, to_currency: str) -> dict:
    rate = 1.1 if from_currency.upper() == "EUR" else 0.9
    return {
        "status": "success",
        "converted_amount": round(float(amount) * rate, 2),
        "rate": rate,
        "from_currency": from_currency,
        "to_currency": to_currency,
    }


def modify_autopay(bill_type: str, source_account: str) -> dict:
    return {
        "status": "success",
        "autopay_enabled": True,
        "bill": bill_type,
        "source": source_account,
    }


# ── Housing & Location ──────────────────────────────────────────
def search_apartments(
    city: str | None = None,
    bedrooms: int | None = None,
    max_price: float | None = None,
    pets_allowed: bool | None = None,
) -> dict:
    # Every filter is optional: the user may give only some of them, and the
    # LLM must not have to invent the rest.
    apartment = {
        "id": "APT1",
        "address": f"12 Oak Street, {city}" if city else "12 Oak Street",
        "price": max_price - 100 if max_price else 1450.0,
        "beds": bedrooms if bedrooms is not None else 1,
    }
    if pets_allowed is not None:
        apartment["pets_allowed"] = pets_allowed
    return {"status": "success", "city": city, "results": [apartment]}


def calculate_commute(
    origin_address: str, destination_address: str, mode: str = "driving"
) -> dict:
    return {"status": "success", "duration_mins": 25, "mode": mode}


def update_search_filter(filter_name: str, value: Any) -> dict:
    return {"status": "success", "filter_updated": filter_name, "new_value": value}


# Housing argument normalisation: spoken values arrive as "$1,800", "3" or
# "yes"; turn them into the types the API expects, or reject them with a
# message the LLM can correct.
_AMOUNT = re.compile(
    r"[$₹€£]?\s*(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*"
    r"(?:dollars?|usd|rupees?|inr|euros?|pounds?)?\s*"
    r"(?:per month|a month|/\s*month|monthly)?",
    re.I,
)


def _price(field: str, value: Any) -> float | int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = value
    elif isinstance(value, str) and (m := _AMOUNT.fullmatch(value.strip())):
        number = float(m.group(1).replace(",", "") + (m.group(2) or ""))
    else:
        raise ToolValidationError(
            f"{field} must be a plain number such as 1800 (for '$1,800'); got {value!r}"
        )
    if number <= 0:
        raise ToolValidationError(f"{field} must be greater than 0; got {value!r}")
    return int(number) if float(number).is_integer() else number


def _count(field: str, value: Any) -> int:
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ToolValidationError(
            f"{field} must be a whole number of bedrooms such as 2; got {value!r}"
        )
    return value


def _flag(field: str, value: Any) -> bool:
    if isinstance(value, str) and value.strip().lower() in ("true", "yes"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no"):
        return False
    if not isinstance(value, bool):
        raise ToolValidationError(f"{field} must be true or false; got {value!r}")
    return value


def _text(field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolValidationError(f"{field} must be a non-empty name; got {value!r}")
    return value.strip()


_SEARCH_FIELDS = {
    "city": _text,
    "bedrooms": _count,
    "max_price": _price,
    "pets_allowed": _flag,
}
_FILTERS = {
    "neighborhood": _text,
    "min_bedrooms": _count,
    "max_price": _price,
    "pets_allowed": _flag,
}


def _prepare_search(args: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(
        k for k, v in args.items() if k not in _SEARCH_FIELDS and v is not None
    )
    if unknown:
        raise ToolValidationError(
            f"unknown argument(s) {unknown}; search_apartments accepts only "
            f"{list(_SEARCH_FIELDS)}"
        )
    return {k: _SEARCH_FIELDS[k](k, v) for k, v in args.items() if v is not None}


def _prepare_filter(args: dict[str, Any]) -> dict[str, Any]:
    name = args.get("filter_name")
    if name not in _FILTERS:
        raise ToolValidationError(
            f"filter_name must be one of {list(_FILTERS)}; got {name!r}"
        )
    if args.get("value") is None:
        raise ToolValidationError(f"value is required for filter {name!r}")
    return {
        "filter_name": name,
        "value": _FILTERS[name](f"value for {name}", args["value"]),
    }


# ── E-Commerce Support ──────────────────────────────────────────
def track_order(order_id: str) -> dict:
    return {
        "status": "success",
        "order_id": order_id,
        "shipping_status": "Out for delivery",
    }


def search_products(query: str, max_price: float | None = None) -> dict:
    price = max_price - 10 if max_price else 99.99
    return {
        "status": "success",
        "products": [
            {"product_id": "PROD1", "name": f"{query} Premium", "price": price}
        ],
    }


def add_to_cart(product_id: str, quantity: int = 1) -> dict:
    return {
        "status": "success",
        "product_id": product_id,
        "quantity": quantity,
        "cart_total": round(99.99 * quantity, 2),
    }


def _obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


_STR = {"type": "string"}
_NUM = {"type": "number"}
_INT = {"type": "integer"}

# (handler, kind, description, parameters)
_SPECS: list[tuple[Any, str, str, dict[str, Any]]] = [
    (
        search_flights,
        "read_only",
        "Search available flights to a destination on a date. Use it whenever the "
        "user asks about flights, fares or flight options.",
        _obj(
            {
                "destination": _STR,
                "date": {**_STR, "description": "Travel date, e.g. 2026-10-02"},
                "origin": _STR,
            },
            ["destination", "date"],
        ),
    ),
    (
        book_flight,
        "state_modifying",
        "Book a flight for a passenger. Only call this once the user has clearly "
        "asked to book; never call it for price or availability questions.",
        _obj(
            {"passenger_name": _STR, "flight_id": {**_STR, "default": "FL123"}},
            ["passenger_name"],
        ),
    ),
    (
        update_identity_doc,
        "state_modifying",
        "Update the user's identity document details (simulated environment). "
        "Call it when the user asks to update a passport, licence or ID number.",
        _obj({"doc_type": _STR, "doc_number": _STR}, ["doc_type", "doc_number"]),
    ),
    (
        get_card_benefits,
        "read_only",
        "Look up the benefits of a credit card type. Always use this instead of "
        "answering from memory.",
        _obj({"card_type": _STR}, ["card_type"]),
    ),
    (
        get_exchange_rate,
        "read_only",
        "Get the exchange rate and converted amount between two currencies. "
        "Always use this instead of estimating a rate.",
        _obj(
            {"amount": _NUM, "from_currency": _STR, "to_currency": _STR},
            ["amount", "from_currency", "to_currency"],
        ),
    ),
    (
        modify_autopay,
        "state_modifying",
        "Change the account that pays a bill automatically (autopay).",
        _obj(
            {"bill_type": _STR, "source_account": _STR},
            ["bill_type", "source_account"],
        ),
    ),
    (
        search_apartments,
        "read_only",
        "Search rental apartments. Every filter is optional: pass only what the "
        "user actually said and omit the rest. Never invent a city, bedroom "
        "count, budget or pet requirement, and search right away instead of "
        "asking for filters the user didn't give (a search without a city is "
        "fine).",
        _obj(
            {
                "city": {**_STR, "description": "City the user named"},
                "bedrooms": {**_INT, "description": "Number of bedrooms"},
                "max_price": {
                    **_NUM,
                    "description": "Maximum monthly rent as a plain number, "
                    "e.g. 1800 for '$1,800' or 'eighteen hundred'",
                },
                "pets_allowed": {
                    "type": "boolean",
                    "description": "true only if the user needs a pet-friendly place",
                },
            },
            [],
        ),
    ),
    (
        calculate_commute,
        "read_only",
        "Calculate the commute time between two addresses. Always use this "
        "instead of estimating. A place name is a valid address: pass it as the "
        "user said it (e.g. 'Downtown Office', 'my office', 'the train station') "
        "instead of asking for a street address.",
        _obj(
            {
                "origin_address": _STR,
                "destination_address": _STR,
                "mode": {
                    **_STR,
                    "enum": ["driving", "transit", "walking", "cycling"],
                    "default": "driving",
                },
            },
            ["origin_address", "destination_address"],
        ),
    ),
    (
        update_search_filter,
        "state_modifying",
        "Update one of the user's saved apartment-search filters. Call it once "
        "per filter that changes.",
        _obj(
            {
                "filter_name": {
                    **_STR,
                    "enum": list(_FILTERS),
                    "description": "neighborhood: area name; min_bedrooms: "
                    "minimum bedrooms; max_price: maximum monthly rent; "
                    "pets_allowed: pet-friendly only",
                },
                "value": {
                    "type": ["number", "boolean", "string"],
                    "description": "A number for max_price and min_bedrooms "
                    "(1800, 3), true/false for pets_allowed, text for neighborhood",
                },
            },
            ["filter_name", "value"],
        ),
    ),
    (
        track_order,
        "read_only",
        "Get the shipping status of an order. Call it for every order ID the "
        "user asks about.",
        _obj({"order_id": _STR}, ["order_id"]),
    ),
    (
        search_products,
        "read_only",
        "Search products, optionally under a maximum price. Use it whenever the "
        "user asks for product recommendations.",
        _obj({"query": _STR, "max_price": _NUM}, ["query"]),
    ),
    (
        add_to_cart,
        "state_modifying",
        "Add a product to the user's shopping cart.",
        _obj(
            {"product_id": _STR, "quantity": {**_INT, "default": 1}},
            ["product_id"],
        ),
    ),
]


_PREPARE = {
    "search_apartments": _prepare_search,
    "update_search_filter": _prepare_filter,
}


def register_mock_apis(
    registry: ToolRegistry, injector: LatencyInjector | None = None
) -> ToolRegistry:
    for handler, kind, description, parameters in _SPECS:
        registry.register(
            ToolSpec(
                name=handler.__name__,
                description=description,
                kind=kind,  # type: ignore[arg-type]
                parameters=parameters,
                handler=handler,
                latency=injector.inject if injector else None,
                prepare=_PREPARE.get(handler.__name__),
            )
        )
    return registry
