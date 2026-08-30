# Financial Crew — Multi-Agent Financial Analysis System

A zero-cost, multi-agent financial analysis system built with a neuro-symbolic
architecture: LLM-based agents handle language and reasoning, while
deterministic tools (yfinance, DuckDuckGo search via `ddgs`) and a validation
agent enforce factual grounding and prevent hallucinated data.

## Overview

The system produces a structured JSON report (price, trend, news sentiment)
for a given asset, combining a news-sentiment pass, a technical-data pass,
and a validation/consolidation pass, each handled by a dedicated CrewAI
agent. It runs entirely on free-tier services: Groq for LLM inference,
`yfinance` for market data, and `ddgs` for news search — no paid APIs are
required.

Supporting infrastructure: a Flask dashboard for triggering and viewing
analyses, a SQLite-backed result cache with TTL, a token-bucket rate limiter
tuned to Groq's free-tier limits, a flat-file metrics log, a pytest suite
with mocked network/LLM calls, GitHub Actions CI, and a Docker/Docker
Compose setup.

## Architecture

Three CrewAI agents run in a sequential pipeline:

- News Analyst — retrieves recent news headlines for an asset via DuckDuckGo
  search (`ddgs`) and produces a sentiment summary grounded strictly in the
  retrieved headlines.
- Technical Analyst — retrieves current price and 1-month history via
  `yfinance` and produces an objective technical summary.
- Orchestrator / Critic — has no tools of its own. Its sole responsibility is
  to validate the outputs of the two agents above, reject any figure or
  claim without traceable provenance, and consolidate everything into a
  single JSON report. It runs at `temperature=0.0`.

Design principle: LLMs are subordinated to deterministic tools and a
validation layer. The Orchestrator does not generate new facts — it audits
and formats. Tool-level errors are typed (`InvalidTickerError`,
`MarketDataUnavailableError`, `NewsSearchError`, `LLMConfigError`) instead of
generic exceptions; network/rate-limit errors are retried with exponential
backoff (`tenacity`), data errors (e.g. an invalid ticker) are not.

Rather than relying on CrewAI's `output_pydantic` for structured output,
`crew.py` extracts the JSON object from the Orchestrator's raw text output
with a regex and validates it manually against `schemas.RelatorioFinanceiro`
(`_extract_json_report`) — see Limitations for why.

```
financial_crew/
├── llm_config.py          # LLM configuration and connectivity test (Groq)
├── agents.py               # Agent definitions (News Analyst, Technical Analyst, Orchestrator)
├── tasks.py                 # Task definitions bound to each agent
├── crew.py                  # Crew assembly, execution, cache lookup, manual JSON extraction
├── schemas.py                # Pydantic schema for the final report
├── cache.py                   # SQLite + TTL cache of analysis results
├── rate_limiter.py             # Token bucket rate limiter for Groq's free tier
├── metrics.py                   # Observability: duration/token metrics per analysis
├── main.py                       # Production entrypoint (CLI, batch execution, error isolation)
├── frontend/app.py                # Flask dashboard (triggers analyses, live-updating UI)
├── tests/                          # pytest suite, network and LLM mocked
├── tools/                           # yfinance / ddgs business logic + CrewAI tool adapters
├── finetuning/                       # optional: LoRA fine-tuned sentiment classifier (see below)
└── docs/AZURE-DEPLOYMENT.md            # documented (not executed) Azure Container Apps deployment
```

## Setup

Requirements: Python 3.10+, and a free Groq API key
(https://console.groq.com/keys). No paid APIs are used.

```bash
python -m venv venv
# Linux / macOS: source venv/bin/activate
# Windows (PowerShell): .\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

Note: `llama-3.3-70b-versatile` was deprecated by Groq on 2026-06-17;
`openai/gpt-oss-120b` is the current default. Check
https://console.groq.com/docs/models if you encounter a "model
decommissioned" error in the future.

### Docker (alternative)

```bash
docker compose up --build
```

Runs the Flask dashboard at `http://localhost:5000`. `output/`, `metrics/`,
and `cache/` are mounted as volumes and persist across restarts. The batch
CLI runs as a separate profile instead of starting automatically:

```bash
docker compose run --rm cli
docker compose run --rm cli python main.py --ticker Bitcoin:BTC-USD
```

## Usage

### Dashboard

```bash
python frontend/app.py
```

Open `http://localhost:5000`, select from the available assets (Apple,
Tesla), and run an analysis. Jobs run sequentially on a single background
worker (to avoid doubling up on Groq's rate limit); the page polls
`/api/status` every 4 seconds. A full analysis takes 1-2 minutes per asset
because of the rate-limit pauses described in Limitations. Reports already
on disk are loaded on startup.

### Batch CLI

```bash
python main.py                                          # default: Apple, Tesla
python main.py --ticker Bitcoin:BTC-USD --ticker MSFT:MSFT
```

Prints a per-asset status (`SUCCESS`, `LLM_ERROR`, `TOOL_ERROR`,
`INVALID_SCHEMA`, `RATE_LIMIT_ERROR`, `UNEXPECTED_ERROR`) and exits `0` only
if all assets succeeded. Reports are saved as JSON to
`output/<TICKER>_<timestamp>.json`.

### Programmatic

```python
from crew import run_analysis

result = run_analysis(asset_name="Tesla", ticker="TSLA")
# or, bypassing the cache:
result = run_analysis(asset_name="Tesla", ticker="TSLA", use_cache=False)
```

### Optional: fine-tuned sentiment classifier

`finetuning/` is a self-contained, optional module: it trains a sentiment
classifier (LoRA over `distilbert-base-uncased`) as an alternative to the
News Analyst's default LLM-based classification, tracks experiments with
MLflow, and serves the result via a separate FastAPI + Docker service. It
adds no dependency to the main crew unless you opt in.

```bash
# 1. train (separate deps, see finetuning/README.md)
pip install -r finetuning/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
python -m finetuning.run_experiments
python -m finetuning.compare --n-per-class 20   # measured comparison vs. the LLM path

# 2. serve (own Dockerfile, own requirements.txt -- see finetuning/README.md)
docker build -f finetuning/serving/Dockerfile -t financial-crew-sentiment .
docker run -p 8001:8001 financial-crew-sentiment

# 3. opt the News Analyst into it
SENTIMENT_PROVIDER=finetuned python main.py --ticker Apple:AAPL
```

`SENTIMENT_PROVIDER` (default: `llm`, i.e. unchanged original behavior)
and `SENTIMENT_SERVICE_URL` (default: `http://localhost:8001`) are the
only two things this integration touches in the main crew. See
`finetuning/README.md` for full usage, `finetuning/RESULTS.md` for
measured accuracy/latency/cost numbers, and `docs/AZURE-DEPLOYMENT.md`
for a documented (not executed) cloud deployment path.

### Tests, metrics, cache

```bash
pip install -r requirements-dev.txt
pytest tests/ -v --cov=. --cov-report=term-missing

python metrics.py   # summary of recorded runs (metrics/metrics.jsonl)
python cache.py     # inspect what's currently cached (cache/analysis_cache.db)
```

CI (`.github/workflows/ci.yml`) runs on every push/PR, on Python 3.11 and
3.12: `ruff check`, `black --check`, and the full test suite with coverage.
No real Groq calls happen in CI; a placeholder `GROQ_API_KEY` is injected
only so modules that check for its presence at import time don't error out.

## Tech Stack

- CrewAI + `litellm` — agent orchestration and LLM connectivity (Groq)
- Groq (`openai/gpt-oss-120b`) — LLM inference, free tier
- `yfinance`, `ddgs` — market data and news search, both free and unauthenticated
- Flask — dashboard frontend
- Pydantic — output schema validation
- SQLite — analysis result cache
- `tenacity` — retry with exponential backoff
- pytest, `pytest-cov` — testing, with network/LLM mocked
- ruff, black — lint and format check
- GitHub Actions — CI
- Docker, Docker Compose
- (optional, `finetuning/`) PyTorch, Hugging Face `transformers`/`peft`/`datasets`,
  MLflow, FastAPI — LoRA fine-tuning, experiment tracking, and serving of an
  alternative sentiment classifier; see `finetuning/README.md`

## Limitations

- Groq's `gpt-oss` models unreliably support tool-call-forced structured
  output (`output_pydantic` / `instructor`): the model sometimes returns
  correct JSON as plain text instead of a formal tool call, which Groq's API
  rejects. The project works around this with manual regex extraction and
  schema validation instead of relying on forced structured output.
- A crewAI bug (crewAIInc/crewAI#5886, open as of writing) injects a
  `cache_breakpoint` field that Groq's API rejects; `llm_config.py`
  monkey-patches it at import time (marked with a `TODO` for removal once
  crewAI ships an official fix).
- The `openai/gpt-oss-120b` model is capped at 8000 tokens/minute and
  200,000 tokens/day on Groq's free tier. A single analysis makes 3 LLM
  calls in quick succession, which can saturate the per-minute budget.
  Pacing is handled entirely by an in-process token bucket
  (`rate_limiter.py`, capacity 8000, refill period 60s, ~1800 tokens
  reserved per call, reconciled against actual usage after each run) — it
  does not coordinate across separate OS processes, so it cannot prevent
  the daily cap from being hit if multiple processes share the same Groq
  account. A retry-with-backoff safety net in `crew.py` covers the cases
  the token bucket can't fully prevent.
- The default asset list is trimmed to 2 (Apple, Tesla) because running the
  full original set repeatedly during development exhausted the daily
  token budget.
- Metrics are recorded to a flat JSONL file (`metrics/metrics.jsonl`), not a
  real time-series backend; adequate for a personal/demo project.
- The analysis cache (SQLite) and rate limiter (in-process token bucket) are
  both local to a single machine/process group — there is no coordination
  mechanism (e.g. Redis) for multiple deployed instances sharing one Groq
  account.
- Tests are unit-level only, with `yfinance`, `ddgs`, and the LLM fully
  mocked; there is no integration test exercising the real Groq API or live
  data sources.
- Possible extensions beyond current scope: WebSocket-based live updates
  instead of polling, persistence to a database instead of flat JSON files,
  and a scheduled/cron execution mode for periodic reports.
- `litellm` (an unpinned dependency of `crewai`) has, as of recent
  releases, started requiring Python 3.11+ (`typing.NotRequired`). If your
  local virtualenv was created with Python 3.10, real LLM calls will fail
  with an `ImportError` even though the package installs cleanly — this
  doesn't affect CI (Python 3.11/3.12) or any mocked test, only a real
  `.venv` built on 3.10. Recreate the virtualenv with Python 3.11+ if you
  hit this.
