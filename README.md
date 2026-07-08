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
| 5     | Production output (strict schema)         | Pending   |

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
├── .env.example
├── llm_config.py          # LLM configuration and connectivity test (Groq)
├── agents.py               # Agent definitions (News Analyst, Technical Analyst, Orchestrator)
├── tasks.py                 # Task definitions bound to each agent
├── crew.py                  # Crew assembly and execution entrypoint
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

Copy the template and fill in your real key:

```bash
cp .env.example .env       # Windows: copy .env.example .env
```

Edit `.env`:

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

## Known Issues

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

## Roadmap

Block 5 will enforce a strict Pydantic schema on the Orchestrator's final
output, wrap execution in a robust `main` entrypoint, and make the result
easily consumable by a frontend or database.
