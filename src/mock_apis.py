"""Simulated Travel, Finance, Housing and E-commerce APIs.

The handlers are pure functions: calling one is the "commit". Simulated network
latency is applied separately (see ``LatencyInjector``) so the coordinator can
re-check the user's intent between the delay and the commit of a mutation.
"""

from __future__ import annotations

import itertools
from typing import Any

from latency_injector import LatencyInjector
from tool_registry import ToolRegistry, ToolSpec

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
def search_apartments(city: str, bedrooms: int, max_price: float) -> dict:
    return {
        "status": "success",
        "city": city,
        "results": [{"id": "APT1", "price": max_price - 100, "beds": bedrooms}],
    }


def calculate_commute(
    origin_address: str, destination_address: str, mode: str = "driving"
) -> dict:
    return {"status": "success", "duration_mins": 25, "mode": mode}


def update_search_filter(filter_name: str, value: str) -> dict:
    return {"status": "success", "filter_updated": filter_name, "new_value": value}


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
        "Search rental apartments in a city by bedroom count and maximum price.",
        _obj(
            {"city": _STR, "bedrooms": _INT, "max_price": _NUM},
            ["city", "bedrooms", "max_price"],
        ),
    ),
    (
        calculate_commute,
        "read_only",
        "Calculate the commute time between two addresses. Always use this "
        "instead of estimating.",
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
        "Update one of the user's saved search filters (e.g. budget, bedrooms).",
        _obj({"filter_name": _STR, "value": _STR}, ["filter_name", "value"]),
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
            )
        )
    return registry
