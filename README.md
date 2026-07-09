# Financial Crew — Multi-Agent Financial Analysis System

A zero-cost, multi-agent financial analysis system built with a neuro-symbolic
architecture: LLM-based agents handle language and reasoning, while
deterministic tools (yfinance, DuckDuckGo search) and a validation agent
enforce factual grounding and prevent hallucinated data.

## Status

| Block | Description                              | Status    |
|-------|-------------------------------------------|-----------|
| 1     | Environment setup and deterministic tools | Done      |
| 2     | LLM configuration (Groq)                  | Done      |
| 3     | Agent definitions                         | Done      |
| 4     | Task and Crew assembly                    | Done      |
| 5     | Production output (strict schema)         | Done      |
| —     | Frontend dashboard (Flask)                | Done      |
| —     | Automated tests (pytest)                  | Done      |
| —     | CI/CD (GitHub Actions)                    | Done      |
| —     | Observability & metrics                   | Done      |

## Architecture

Three CrewAI agents run in a sequential pipeline:

1. **News Analyst** — retrieves recent news headlines for an asset via
   DuckDuckGo search (`ddgs`) and produces a sentiment summary grounded
   strictly in the retrieved headlines.
2. **Technical Analyst** — retrieves current price and 1-month history via
   `yfinance` and produces an objective technical summary.
3. **Orchestrator / Critic** — has no tools of its own. Its sole
   responsibility is to validate the outputs of the two agents above,
   reject any figure or claim without traceable provenance, and consolidate
   everything into a single JSON report. It runs at `temperature=0.0`.

Design principle: LLMs are subordinated to deterministic tools and a
validation layer. The Orchestrator does not generate new facts — it audits
and formats.

## Project Structure

```
financial_crew/
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .coveragerc
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions: lint, format check, tests (on push/PR)
├── llm_config.py          # LLM configuration and connectivity test (Groq)
├── agents.py               # Agent definitions (News Analyst, Technical Analyst, Orchestrator)
├── tasks.py                 # Task definitions bound to each agent
├── crew.py                  # Crew assembly, execution, manual JSON extraction/validation
├── schemas.py                # Pydantic schema for the final report
├── metrics.py                 # Observability: duration/token metrics per analysis
├── main.py                     # Production entrypoint (CLI, batch execution, error isolation)
├── frontend/
│   └── app.py                   # Flask dashboard (triggers analyses, live-updating UI)
├── tests/
│   ├── conftest.py
│   ├── test_market_data_tool.py
│   ├── test_news_tool.py
│   ├── test_schemas.py
│   ├── test_crew_json_extraction.py
│   └── test_metrics.py
└── tools/
    ├── __init__.py
    ├── exceptions.py         # Custom exception hierarchy
    ├── market_data_tool.py    # yfinance business logic
    ├── news_tool.py            # ddgs business logic
    └── crew_tools.py            # CrewAI BaseTool adapters
```

## Requirements

- Python 3.10+
- A free Groq API key: https://console.groq.com/keys

No paid APIs are used. Data sources: `yfinance` (market data) and `ddgs`
(news search), both free and unauthenticated.

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

```bash
# Linux / macOS
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If you are on Windows and see an error related to `uvloop` ("uvloop does
not support Windows at the moment"), it means a package pulled in an
optional dependency that is Linux/macOS-only. This should not happen with
the pinned versions in `requirements.txt`, but if it does, install the
core dependencies individually first (`pip install yfinance ddgs pydantic
tenacity crewai litellm python-dotenv`) and report which package triggered
it.

### 3. Configure the Groq API key

Create a file named `.env` in the project root with:

```
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

Note: `llama-3.3-70b-versatile` was deprecated by Groq on 2026-06-17. The
current default, `openai/gpt-oss-120b`, is the recommended replacement.
Check https://console.groq.com/docs/models if you encounter a "model
decommissioned" error in the future.

## Validating Each Block

Each script can be run standalone to validate that block in isolation,
without needing the rest of the pipeline to be complete.

### Block 1 — Tools

```bash
python tools/market_data_tool.py
python tools/news_tool.py
```

Expected: a JSON block with real market/news data for the sample inputs,
and a clean, typed error (not a traceback) for the invalid-input cases.

### Block 2 — LLM connectivity

```bash
python llm_config.py
```

Expected output:

```
--- Teste de ligação ao LLM (Groq) ---
LLM respondeu corretamente: 'OK'
```

### Block 3 — Agent construction

```bash
python agents.py
```

Expected: confirmation that all three agents are built without errors.
This does not call the LLM.

### Block 4 — Crew assembly

```bash
python crew.py
```

Expected: confirmation that the Crew is assembled with 3 agents and 3
tasks in the correct order. This does not execute the crew (no LLM or API
calls are made by this script).

To run a full, real analysis (this does call the Groq API and the data
tools, consuming API quota):

```python
from crew import run_analysis

result = run_analysis(asset_name="Tesla", ticker="TSLA")
print(result)
```

Supported sample assets for this project: Apple (AAPL), Tesla (TSLA),
Bitcoin (BTC-USD), Ethereum (ETH-USD), Microsoft (MSFT).

## Error Handling Philosophy

All tool-level errors are typed (`InvalidTickerError`,
`MarketDataUnavailableError`, `NewsSearchError`, `LLMConfigError`) rather
than generic exceptions, so that failures are distinguishable and can be
surfaced explicitly instead of silently producing incomplete or
fabricated data. Network and rate-limit errors are retried automatically
with exponential backoff (via `tenacity`); data errors (e.g. an invalid
ticker) are not retried, since retrying will not change the outcome.

## Implementation Notes

### crewAI + Groq: "cache_breakpoint is unsupported"

crewAI versions 1.14.4+ have an open bug ([crewAIInc/crewAI#5886](https://github.com/crewAIInc/crewAI/issues/5886)) where a `cache_breakpoint` field is injected into every message sent to the LLM, but only the Anthropic adapter knows how to strip it before the API call. Groq (and other OpenAI-compatible providers) reject this field with:

```
litellm.BadRequestError: GroqException - {"error":{"message":"'messages.0' :
for 'role:system' the following must be satisfied[('messages.0' : property
'cache_breakpoint' is unsupported)]"}}
```

`llm_config.py` applies the workaround suggested in the issue thread: a
monkey-patch that no-ops `crewai.llms.cache.mark_cache_breakpoint`. This is
applied automatically at import time — no action needed on your part. Once
crewAI ships an official fix, this patch (clearly marked with a `TODO` in
`llm_config.py`) can be removed.

### Block 5 — Strict schema and production entrypoint

`schemas.py` defines the exact Pydantic structure (`RelatorioFinanceiro`)
the Orchestrator's output must satisfy. Rather than forcing this via
CrewAI's `output_pydantic` (which relies on tool-call-forced structured
output — unreliable on Groq's `gpt-oss` models, see "Known Issues"
below), `crew.py` extracts the JSON object from the Orchestrator's raw
text output with a regex and validates it manually against the schema
(`_extract_json_report`). This is both more robust and cheaper (one
fewer LLM round-trip per analysis).

`main.py` wraps everything in a batch-friendly, fault-isolated entrypoint:
each asset is analyzed independently (one failure does not abort the
others), validated reports are saved to `output/<TICKER>_<timestamp>.json`,
and a summary with per-asset status is printed at the end.

Run the default set of assets (Apple, Tesla):

```bash
python main.py
```

Run specific assets only:

```bash
python main.py --ticker Bitcoin:BTC-USD --ticker Microsoft:MSFT
```

Expected output: a per-asset status line (`SUCCESS`, `LLM_ERROR`,
`TOOL_ERROR`, `INVALID_SCHEMA`, `RATE_LIMIT_ERROR`, or
`UNEXPECTED_ERROR`), a summary count, and a process exit code of `0` if
all assets succeeded or `1` otherwise (useful for CI/scripting).
Successful reports are written as JSON files under `output/`.

### Frontend dashboard

`frontend/app.py` is a Flask app that triggers real analyses directly from
the browser -- it does not depend on reports already existing in
`output/`. On load it shows a selection panel with the 2 available assets
(Apple, Tesla); picking one or more and clicking "Analisar selecionados"
enqueues them to a single background worker thread (sequential by design,
to avoid doubling up on Groq's rate limit), and the dashboard polls
`/api/status` every 4 seconds, showing a shimmering skeleton card while
each analysis runs and swapping in the full animated card (price, trend,
sentiment gauge, headlines) the moment it completes. Reports are also
saved to `output/<TICKER>_<timestamp>.json` as a side effect, same as
`main.py`, and any reports already on disk are loaded on startup so the
dashboard has continuity across restarts.

Run it (`main.py` does not need to be running separately -- the frontend
calls the same `crew.run_analysis` directly):

```bash
python frontend/app.py
```

Then open http://localhost:5000 in a browser, select assets, and click
"Analisar selecionados". A full analysis still takes 1-2 minutes per
asset because of the rate-limit pauses described below.

### Groq free tier rate limits

The `openai/gpt-oss-120b` model is capped at 8000 tokens per minute and
200,000 tokens per day on Groq's free tier. A single full analysis makes
3 LLM calls in quick succession (news task, technical task, orchestrator
task) -- this alone can saturate the per-minute budget before the run
even finishes, regardless of any delay *between* assets.

The fix is a deliberate pause *within* each analysis: `tasks.py` attaches
a callback to the news and technical tasks that sleeps for
`SECONDS_BETWEEN_TASKS` (25s by default) after each one completes. This
spreads the 3 LLM calls of a single asset across a wider time window, so
the rolling per-minute limit has room to recover between calls.
`main.py` additionally waits `SECONDS_BETWEEN_ASSETS` (30s) between
assets in a batch run, and `crew.py` retries with exponential backoff as
a last-resort safety net if a per-minute rate limit is still hit.

The **daily** limit (200,000 tokens) is a hard ceiling that no amount of
waiting-within-the-run can work around -- if you exhaust it, you have to
wait for the daily reset (shown in the error message, e.g. "try again in
3m46s" if you're near the boundary, or several hours if you've just
exhausted it) or switch to a different Groq account/model. This is also
why the default asset list was trimmed down to 2 (Apple, Tesla): running
all 5 original assets repeatedly during development exhausted the daily
budget.

If you still see `RATE_LIMIT_ERROR` in the summary, increase
`SECONDS_BETWEEN_TASKS` in `tasks.py` and/or `SECONDS_BETWEEN_ASSETS` in
`main.py`, or check your usage at
https://console.groq.com/settings/billing.

## Known Issues

### Groq tool-call-forced structured output is unreliable

CrewAI's `output_pydantic` (and `instructor`, which it uses internally)
forces the LLM to respond via a tool call so the JSON can be validated
against a schema automatically. On Groq's `gpt-oss` models this is
flaky: the model sometimes emits the correct JSON as plain text instead
of a formal tool call, and Groq's API rejects this with a `400
tool_use_failed` error ("Tool choice is required, but model did not call
a tool") even though the generated content itself is correct and visible
in the error's `failed_generation` field. This is a known ecosystem-wide
issue (see e.g. [pydantic/pydantic-ai#4350](https://github.com/pydantic/pydantic-ai/issues/4350),
which documents the same underlying Groq behavior).

The project works around this by not using `output_pydantic` at all:
`crew.py`'s `_extract_json_report` pulls the JSON object out of the
Orchestrator's plain-text output with a regex and validates it manually
against `schemas.RelatorioFinanceiro`. This is both more reliable and
slightly cheaper (no extra `instructor` round-trip).

## Testing

Unit tests live in `tests/` and never touch the network or the Groq API —
`yfinance`, `ddgs`, and the LLM are all mocked. They cover:

- `tools/market_data_tool.py` and `tools/news_tool.py`: success paths and
  every typed error path (invalid ticker, empty results, rate limits,
  network failures)
- `schemas.py`: valid and invalid `RelatorioFinanceiro` payloads
- `crew._extract_json_report`: the manual JSON extraction/validation logic
  described above, including the realistic "messy" cases (JSON wrapped in
  prose or markdown fences, malformed JSON, schema violations)
- `metrics.py`: recording, reading, and summarizing

Install dev dependencies and run:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v --cov=. --cov-report=term-missing
```

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request, on Python
3.11 and 3.12: installs dependencies, lints with `ruff`, checks formatting
with `black`, and runs the full test suite with coverage. A `GROQ_API_KEY`
placeholder is injected purely so modules that check for its presence at
import/call time (but never actually call the API in tests) don't error
out — no real API calls happen in CI.

## Observability & Metrics

Every call to `crew.run_analysis` — success or failure — is recorded to
`metrics/metrics.jsonl` (one JSON object per line) via `metrics.py`:
duration, token usage (`total`/`prompt`/`completion`/`successful_requests`,
read from CrewAI's `CrewOutput.token_usage`), final status, and the error
message when applicable.

View a summary from the command line:

```bash
python metrics.py
```

Or from the frontend: the dashboard shows a live metrics bar (total runs,
success rate, average duration, total tokens) above the asset cards,
refreshed after every analysis. This is deliberately simple (a flat JSONL
file, no external time-series database) — appropriate for a personal/demo
project, but the schema is structured enough to be loaded into pandas or
a real metrics backend later if needed.

## Roadmap

All five planned blocks are complete, plus a Flask-based frontend
dashboard. Possible next steps beyond the original scope: WebSocket-based
live updates instead of polling, persistence to a database instead of
flat JSON files, and a scheduled/cron
execution mode for periodic reports.
