# Housing Agent Fix — Exact Implementation Instructions for Claude

## Goal

Fix the **Housing domain tool-calling failures** in the agent benchmark without redesigning the whole agent or breaking Ecommerce, Finance, or Travel.

The benchmark currently shows:

- Overall pass rate: **20.6% → 35.4%**
- Correct tools: **36.5% → 66.7%**
- No-tool-call cases: **~half → 14/48**
- Agent talking over user: **88/92 → 1/46**
- Response quality: **12.9% → 56.5%**
- Ecommerce: **29% → 58%**
- Finance: **21% → 67%**
- Travel: **25% → 17%**
- Housing: **17% → 0%**

The Housing failures identified in the benchmark report are:

1. `search_apartments` requires `bedrooms` and `max_price` even when the user did not provide them.
2. The agent invents values such as `max_price: 9999999` or `bedrooms: 1`.
3. The agent uses incorrect filter names such as `preferred_neighborhood` instead of `neighborhood`.
4. The agent uses `bedrooms` instead of `min_bedrooms`.
5. `update_search_filter` accepts arbitrary text for the filter name, allowing the model to invent parameter names.
6. One `$1,800` input became `max_price: 800`.
7. Some Housing scenarios make no tool call because the agent asks a follow-up question instead of acting.

The intended solution is to make the Housing tools **LLM-friendly**, use optional parameters where the user may legitimately omit information, constrain filter names, validate arguments before executing the backend API, and avoid inventing missing values.

---

# IMPORTANT SAFETY / SCOPE RULE

Do NOT rewrite the entire agent.

Do NOT make unrelated changes to Ecommerce, Finance, Travel, benchmark scoring, or the audio/voice pipeline.

Do NOT blindly force a tool call for every user message.

Do NOT invent fake defaults such as:

```json
{"max_price": 9999999}
```

or:

```json
{"bedrooms": 1}
```

when the user did not provide those values.

First inspect the existing repository and understand the current implementation.

Before editing anything, identify:

- the existing Housing tool definitions
- `src/mock_apis.py` or `src/mock_api.py` (whichever actually exists)
- the agent/tool registration code
- `update_search_filter`
- `search_apartments`
- existing tests
- benchmark runner
- current Housing scenarios

Preserve the existing backend API behavior wherever possible. The main change should be the **LLM-facing tool interface and validation layer**.

---

# 1. Inspect the repository first

Search the repository for:

```text
search_apartments
update_search_filter
preferred_neighborhood
min_bedrooms
max_price
bedrooms
Housing
housing
tool_choice
tools=
```

Also locate the exact file corresponding to the benchmark report reference:

```text
src/mock_apis.py:82
```

The filename may be `mock_apis.py` or `mock_api.py`; use the actual repository structure.

Do not assume filenames.

Before making changes, produce a short internal summary of:

1. Current Housing tool schema
2. Current backend API signature
3. Where tools are registered with the LLM
4. How tool results/errors are returned to the model
5. How benchmark scenarios are evaluated

Then implement the changes below.

---

# 2. Make Housing search parameters optional

## Current problem

The LLM-facing `search_apartments` tool currently requires parameters such as:

```text
bedrooms
max_price
```

even when the user did not specify them.

This forces the model to either:

- ask an unnecessary follow-up question, or
- hallucinate a value.

Both hurt the benchmark.

## Desired LLM-facing schema

Use these canonical names:

```text
neighborhood
min_bedrooms
max_price
```

All of these should be optional unless the actual application semantics require otherwise.

Conceptually:

```python
neighborhood: Optional[str] = None
min_bedrooms: Optional[int] = None
max_price: Optional[int] = None
```

If the project uses Pydantic / JSON Schema / function definitions instead of Python type hints, express the same semantics in the appropriate framework.

The resulting tool should allow calls such as:

```json
{
  "neighborhood": "Adyar"
}
```

or:

```json
{
  "min_bedrooms": 2
}
```

or:

```json
{
  "neighborhood": "Adyar",
  "min_bedrooms": 2
}
```

without requiring a fake `max_price`.

If the user gives all three:

```json
{
  "neighborhood": "Adyar",
  "min_bedrooms": 2,
  "max_price": 1800
}
```

that must also work.

---

# 3. Use one canonical vocabulary

The LLM-facing Housing interface must use exactly these names:

```text
neighborhood
min_bedrooms
max_price
```

Do NOT expose:

```text
preferred_neighborhood
bedrooms
```

as alternative names.

The benchmark specifically found:

```text
preferred_neighborhood
```

being sent instead of:

```text
neighborhood
```

and:

```text
bedrooms
```

being sent instead of:

```text
min_bedrooms
```

Do not solve this by merely accepting every possible alias at the LLM layer. The goal is to make the model consistently generate the canonical schema.

If the backend internally uses different names, create a translation in the wrapper.

For example:

```python
def search_housing(
    neighborhood=None,
    min_bedrooms=None,
    max_price=None,
):
    backend_params = {}

    if neighborhood is not None:
        backend_params["neighborhood"] = neighborhood

    if min_bedrooms is not None:
        backend_params["bedrooms"] = min_bedrooms

    if max_price is not None:
        backend_params["max_price"] = max_price

    return search_apartments_backend(**backend_params)
```

Adapt this to the actual repository rather than blindly copying it.

---

# 4. Constrain `update_search_filter`

## Current problem

The benchmark report says:

> `update_search_filter` accepts any text, so the model guesses the name.

If the current schema looks conceptually like:

```python
filter_name: str
```

replace it with a constrained set.

Preferred approach:

```python
from typing import Literal

FilterName = Literal[
    "neighborhood",
    "min_bedrooms",
    "max_price",
]
```

or, if the project uses an enum:

```python
class HousingFilterName(str, Enum):
    NEIGHBORHOOD = "neighborhood"
    MIN_BEDROOMS = "min_bedrooms"
    MAX_PRICE = "max_price"
```

Then the LLM can only select from the supported filter names.

The valid names are:

```text
neighborhood
min_bedrooms
max_price
```

Do not allow arbitrary strings.

---

# 5. Validate Housing tool arguments before backend execution

Add a validation boundary between the LLM and the backend.

Desired flow:

```text
USER
  |
  v
LLM
  |
  | tool call
  v
Housing tool schema
  |
  v
VALIDATION
  |
  +---- valid ----> backend API
  |
  +---- invalid ---> structured error -> LLM can correct call
```

The backend should never receive obviously invalid arguments.

Validate at minimum:

### `neighborhood`

- must be a string if provided
- should not be an empty string

### `min_bedrooms`

- must be an integer if provided
- should be >= 0 or >= 1 according to the existing backend semantics
- do not change the project's intended bedroom semantics without checking existing tests/data

### `max_price`

- must be numeric/integer according to existing API semantics
- must be positive if that is the existing backend requirement
- reject malformed values

### Cross-field validation

If both are present:

```text
min_bedrooms
max_price
```

make sure they are semantically valid according to the existing backend.

Do not introduce arbitrary business rules that don't already exist.

---

# 6. Do NOT invent missing values

This is critical.

If the user says:

> Find apartments in Adyar.

The correct tool call should be approximately:

```json
{
  "neighborhood": "Adyar"
}
```

NOT:

```json
{
  "neighborhood": "Adyar",
  "min_bedrooms": 1,
  "max_price": 9999999
}
```

If the user says:

> I need a 2 bedroom apartment.

Use:

```json
{
  "min_bedrooms": 2
}
```

Do not invent a budget.

If the user says:

> My budget is $1800.

Use:

```json
{
  "max_price": 1800
}
```

Do not turn this into `800`.

Missing optional information should remain missing.

---

# 7. Fix currency/number extraction

The benchmark caught:

```text
$1,800
```

becoming:

```text
800
```

Make sure the Housing argument extraction preserves the full numeric value.

For example:

```text
$1,800 -> 1800
₹1,800 -> 1800
1800 dollars -> 1800
```

The exact supported currencies should follow the existing benchmark/API semantics.

Do not create a large new currency-conversion system unless the repository already requires one.

The immediate requirement is:

```text
1,800 -> 1800
```

not:

```text
800
```

Add a regression test for this exact case.

---

# 8. Add an LLM-friendly wrapper if the backend API is rigid

If the current backend API requires parameters that the user may not provide, do NOT expose that rigid backend signature directly to the LLM.

Use a wrapper/facade.

Conceptually:

```python
def search_housing(
    neighborhood=None,
    min_bedrooms=None,
    max_price=None,
):
    # validate
    # translate into backend parameters
    # call existing backend
    # return existing backend result
```

The wrapper is the LLM-facing interface.

The existing backend can remain more rigid if necessary.

This separation is intentional:

```text
LLM-friendly interface
        |
        v
validation / normalization
        |
        v
existing backend API
```

This minimizes the risk of breaking existing behavior.

---

# 9. Improve the Housing tool description

The tool description should explicitly tell the model:

- use only the supported parameter names
- omit parameters the user did not provide
- never invent a budget
- never invent bedroom count
- use `min_bedrooms`, not `bedrooms`
- use `neighborhood`, not `preferred_neighborhood`
- use `max_price` for the user's maximum budget

Conceptually:

```text
Search for housing using only information provided by the user.

Supported filters:
- neighborhood: preferred neighborhood
- min_bedrooms: minimum number of bedrooms
- max_price: maximum price/budget

All filters are optional.

Do not invent missing values.
If the user does not provide a filter, omit it from the tool call.

Use exactly these parameter names:
neighborhood
min_bedrooms
max_price
```

Adapt wording to the project's current tool-definition format.

Do not overload the description with unnecessary instructions.

---

# 10. Do NOT globally force tool calls

The benchmark had many "no tool call" cases.

Do not solve this by globally forcing:

```text
tool_choice = "required"
```

for every user message.

That can cause nonsense tool calls for messages that don't require tools.

Instead:

- keep normal tool-selection behavior
- make the Housing tool usable when sufficient information is available
- make optional fields actually optional
- improve the tool description
- preserve normal conversational behavior when no action is needed

If the existing architecture has explicit scenario routing, use it rather than forcing every message into a tool call.

---

# 11. Add a correction path for invalid arguments

If the LLM still produces an invalid tool call, do not silently execute bad data.

Return a structured validation error that the model can understand.

Conceptually:

```json
{
  "error": "invalid_tool_arguments",
  "field": "max_price",
  "message": "max_price must be an integer. The user provided '$1,800'."
}
```

The exact implementation should match the project's current tool execution architecture.

The important behavior is:

```text
invalid call
    ↓
validation error
    ↓
LLM correction
    ↓
valid call
    ↓
backend execution
```

Do not expose stack traces to the user.

---

# 12. Tests to add

Add focused tests before running the full benchmark.

## Test 1 — Neighborhood only

Input:

```text
Find me an apartment in Adyar.
```

Expected Housing tool arguments:

```json
{
  "neighborhood": "Adyar"
}
```

Must NOT contain invented:

```text
min_bedrooms
max_price
```

---

## Test 2 — Bedroom only

Input:

```text
I need at least 2 bedrooms.
```

Expected:

```json
{
  "min_bedrooms": 2
}
```

Must NOT contain:

```text
bedrooms
```

or invented budget.

---

## Test 3 — Budget only

Input:

```text
My maximum budget is $1,800.
```

Expected:

```json
{
  "max_price": 1800
}
```

Explicitly verify:

```text
1800 != 800
```

---

## Test 4 — All filters

Input:

```text
Find me a 2 bedroom apartment in Adyar under $1,800.
```

Expected:

```json
{
  "neighborhood": "Adyar",
  "min_bedrooms": 2,
  "max_price": 1800
}
```

---

## Test 5 — Invalid filter name

Attempt:

```json
{
  "filter_name": "preferred_neighborhood"
}
```

Expected:

```text
validation failure
```

The system should not silently accept it as a valid canonical filter.

---

## Test 6 — Invalid bedroom parameter

Attempt:

```json
{
  "bedrooms": 2
}
```

At the LLM-facing schema level this should not be a valid canonical parameter.

The canonical field is:

```text
min_bedrooms
```

---

## Test 7 — Missing optional values

Input:

```text
Show me apartments.
```

Do not generate fake:

```text
max_price
min_bedrooms
neighborhood
```

Follow the existing benchmark's intended behavior for whether this request should search or ask for clarification. The key requirement is: **never invent values just because the backend previously required them**.

---

# 13. Run existing tests

Before changing benchmark behavior, run the project's existing test suite.

Use the repository's normal command, for example:

```bash
pytest
```

if that is what the project uses.

Do not assume pytest if the project uses another test runner.

Record:

```text
before changes:
X passed / Y failed
```

Then after changes:

```text
after changes:
X passed / Y failed
```

Do not delete or weaken existing tests to make them pass.

---

# 14. Run the Housing-focused tests

Run the newly added tests separately if possible.

Confirm:

- optional fields work
- canonical names work
- invalid names are rejected
- `$1,800` becomes `1800`
- missing values are not fabricated
- backend behavior remains intact

---

# 15. Run the full 48-sample benchmark

After focused tests pass, run the same benchmark used to produce the report.

Do NOT change:

- benchmark scenarios
- scoring rules
- judge model
- pass criteria

unless absolutely necessary for a genuine bug in the benchmark itself.

Use the same setup so the comparison is meaningful.

Record:

```text
Overall pass rate
Correct tools
No tool call
Agent talked over user
Response quality
Median first reply
Ecommerce
Finance
Travel
Housing
```

The main metric to inspect is:

```text
Housing
```

but also make sure the other domains did not regress.

---

# 16. Success criteria

The changes are successful if:

### Functional

- Housing tools accept omitted optional fields.
- The LLM no longer needs to invent `max_price` or bedroom count.
- Canonical parameter names are used.
- Invalid filter names are rejected.
- `$1,800` is parsed as `1800`.
- Invalid tool arguments cannot silently reach the backend.

### Regression safety

Ecommerce, Finance, and Travel should not suffer a significant regression.

The benchmark should show a meaningful improvement in Housing from:

```text
0%
```

toward a substantially higher score.

Do not artificially optimize specifically for a handful of benchmark strings. The implementation should generalize to similar Housing requests.

---

# 17. Important implementation philosophy

Prefer:

```text
better schema
+
better validation
+
clear tool description
+
small wrapper
```

over:

```text
huge system prompt
+
hardcoded benchmark responses
+
fake defaults
+
special cases for individual scenarios
```

Do not hardcode expected benchmark outputs.

Do not create scenario-specific hacks such as:

```python
if user_message == "...":
    return ...
```

The fix must operate at the tool/interface level.

---

# 18. Do not over-engineer

The first implementation should be small.

We are NOT asking for:

- a new agent framework
- a new database
- a new LLM
- a new vector database
- a complete rewrite of the benchmark
- a complex autonomous planner
- a large prompt rewrite

The immediate problem is tool schema quality and validation.

---

# 19. Files likely to change

Only change files that are actually necessary.

Likely candidates:

```text
src/mock_api.py
src/mock_apis.py
agent/tool definition file
Housing tool/schema file
tests/
```

Do not modify both `mock_api.py` and `mock_apis.py` unless both actually participate in the running application.

First determine which file is used.

---

# 20. Final report required from Claude

After implementation, report:

## Files changed

List every changed file.

## What changed

Explain each change briefly.

## Housing schema before

Show the old LLM-facing schema.

## Housing schema after

Show the new LLM-facing schema.

## Validation

Explain how invalid arguments are handled.

## Tests

Report test results.

## Benchmark

Report the new 48-sample benchmark results in this table:

| Metric | Before | After |
|---|---:|---:|
| Overall pass rate | 35.4% | |
| Correct tools | 66.7% | |
| No tool call | 14/48 | |
| Agent talked over user | 1/46 | |
| Response quality | 56.5% | |
| Ecommerce | 58% | |
| Finance | 67% | |
| Travel | 17% | |
| Housing | 0% | |

If the benchmark cannot be run, explicitly say why. Do not fabricate results.

## Regression check

Explicitly state whether Ecommerce, Finance, and Travel improved, stayed roughly stable, or regressed.

---

# 21. Final instruction

Implement the changes now.

First inspect the repository and existing architecture.

Then make the **smallest robust implementation** that solves the Housing tool-calling issues described above.

Do not ask me to manually edit individual lines if you can safely make the changes yourself.

Do not claim success until tests/benchmark results have actually been run.

If something in the repository conflicts with these instructions, inspect the actual code and choose the least invasive implementation that preserves existing behavior.

The core desired architecture is:

```text
                    USER
                      |
                      v
                   LLM
                      |
                      v
             LLM-FRIENDLY TOOL
                      |
          +-----------+-----------+
          |                       |
          v                       v
     VALIDATION              missing optional
          |                    values remain
          |                    omitted
          v
       BACKEND
```

The key rule is:

> **The LLM should only provide information it actually knows from the user. The tool/backend layer should handle translation and validation rather than forcing the LLM to invent missing API parameters.**
