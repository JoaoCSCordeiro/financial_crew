"""
Frontend - Servidor de dashboard.

Serve um dashboard HTML/CSS/JS que permite escolher os ativos a analisar,
despoleta a analise real (crew.run_analysis) num worker em background, e
apresenta os resultados com transicoes suaves e atualizacao automatica --
sem depender de relatorios ja gerados previamente por main.py.

Uso:
    python frontend/app.py
    (depois abre http://localhost:5000 no browser)
"""

import json
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # permite `python frontend/app.py` a partir de qualquer diretoria

from crew import is_rate_limit_error, run_analysis  # noqa: E402
from llm_config import LLMConfigError  # noqa: E402
from schemas import RelatorioFinanceiro  # noqa: E402
from tools.exceptions import FinancialToolError  # noqa: E402

OUTPUT_DIR = BASE_DIR / "output"

# Os 5 ativos do ambito original do projeto. A selecao no frontend e
# limitada a 2 de cada vez (ver limite MAX_SELECTION no JS) para caber
# confortavelmente no free tier da Groq (ver README, "Groq free tier
# rate limits") -- mas qualquer combinacao de 2 pode ser escolhida.
AVAILABLE_ASSETS: dict[str, str] = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Microsoft": "MSFT",
}

app = Flask(__name__)

# --- Estado dos jobs em memoria -------------------------------------------
# Cada ticker tem um estado: "idle" | "queued" | "running" | "success" | "error".
# Um unico worker thread processa a fila sequencialmente, para nunca disparar
# duas analises ao mesmo tempo (o que faria o rate limit da Groq esgotar-se
# ainda mais depressa -- ver README, seccao "Groq free tier rate limits").
_jobs_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_job_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()


def _set_job(ticker: str, **fields) -> None:
    with _jobs_lock:
        _jobs.setdefault(ticker, {}).update(fields)


def _save_report_to_disk(asset_name: str, ticker: str, report: RelatorioFinanceiro) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filepath = OUTPUT_DIR / f"{ticker}_{timestamp}.json"
    filepath.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return filepath


def _worker_loop() -> None:
    """
    Worker unico que consome a fila de pedidos de analise, um de cada vez.
    Corre para sempre numa thread em background (daemon), iniciada quando
    o modulo e importado.
    """
    while True:
        asset_name, ticker = _job_queue.get()
        _set_job(ticker, status="running", asset_name=asset_name, detail=None, stage="noticias")

        def _on_stage_change(stage: str, _ticker=ticker) -> None:
            _set_job(_ticker, stage=stage)

        try:
            result = run_analysis(asset_name=asset_name, ticker=ticker, stage_callback=_on_stage_change)
        except LLMConfigError as exc:
            _set_job(ticker, status="error", detail=f"Erro de configuração do LLM: {exc}")
        except FinancialToolError as exc:
            _set_job(ticker, status="error", detail=f"Erro numa ferramenta de dados: {exc}")
        except Exception as exc:  # noqa: BLE001
            if is_rate_limit_error(exc):
                _set_job(ticker, status="error", detail="Rate limit da Groq excedido. Tenta novamente daqui a alguns minutos.")
            else:
                _set_job(ticker, status="error", detail=f"Erro inesperado: {type(exc).__name__}: {exc}")
        else:
            if isinstance(result, RelatorioFinanceiro):
                _save_report_to_disk(asset_name, ticker, result)
                _set_job(ticker, status="success", report=result.model_dump(), detail=None)
            else:
                _set_job(ticker, status="error", detail="O LLM não devolveu um relatório JSON válido.")
        finally:
            _job_queue.task_done()


def _load_existing_reports() -> None:
    """Preenche o estado inicial dos jobs com relatorios ja gerados anteriormente."""
    if not OUTPUT_DIR.exists():
        return
    latest_by_ticker: dict[str, tuple[str, dict]] = {}
    for filepath in sorted(OUTPUT_DIR.glob("*.json")):
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        ticker = data.get("ticker")
        if ticker:
            latest_by_ticker[ticker] = (data.get("ativo", ticker), data)

    for ticker, (asset_name, data) in latest_by_ticker.items():
        _set_job(ticker, status="success", asset_name=asset_name, report=data, detail=None)


_load_existing_reports()

_worker_thread = threading.Thread(target=_worker_loop, daemon=True)
_worker_thread.start()


@app.route("/api/assets")
def api_assets():
    """Devolve os ativos disponiveis para o frontend construir a seleccao."""
    return jsonify([{"asset_name": name, "ticker": ticker} for name, ticker in AVAILABLE_ASSETS.items()])


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Poe em fila a analise de um ou mais ativos. Body esperado:
    {"tickers": ["AAPL", "TSLA"]}
    Ignora tickers que ja estejam "queued" ou "running", para nao duplicar
    trabalho se o utilizador clicar varias vezes.
    """
    payload = request.get_json(force=True, silent=True) or {}
    tickers = payload.get("tickers", [])

    reverse_lookup = {t: n for n, t in AVAILABLE_ASSETS.items()}
    queued = []
    for ticker in tickers:
        if ticker not in AVAILABLE_ASSETS.values():
            continue
        with _jobs_lock:
            current_status = _jobs.get(ticker, {}).get("status")
        if current_status in ("queued", "running"):
            continue
        asset_name = reverse_lookup[ticker]
        _set_job(ticker, status="queued", asset_name=asset_name, detail=None)
        _job_queue.put((asset_name, ticker))
        queued.append(ticker)

    return jsonify({"queued": queued})


@app.route("/api/status")
def api_status():
    """
    Devolve o estado atual de todos os jobs conhecidos (em memoria), com o
    relatorio completo para os que ja tiverem sucesso.
    """
    with _jobs_lock:
        return jsonify(dict(_jobs))


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="pt-PT">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Financial Crew — Painel de Análise</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0A0E12;
    --surface: #121821;
    --surface-hover: #171F2B;
    --border: #22303F;
    --text: #E8EDF2;
    --text-muted: #6B7785;
    --positive: #34D399;
    --negative: #F87171;
    --mixed: #FBBF24;
    --brand: #5EEAD4;
    --font-display: 'Space Grotesk', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    background: var(--bg);
    background-image:
      radial-gradient(circle at 15% 0%, rgba(94, 234, 212, 0.06), transparent 40%),
      radial-gradient(circle at 85% 100%, rgba(52, 211, 153, 0.05), transparent 40%);
    color: var(--text);
    font-family: var(--font-display);
    min-height: 100vh;
  }

  header {
    padding: 48px 40px 24px;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
    border-bottom: 1px solid var(--border);
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .pulse-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--brand);
    box-shadow: 0 0 0 0 rgba(94, 234, 212, 0.6);
    animation: pulse 2.2s infinite;
  }

  @keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(94, 234, 212, 0.55); }
    70%  { box-shadow: 0 0 0 10px rgba(94, 234, 212, 0); }
    100% { box-shadow: 0 0 0 0 rgba(94, 234, 212, 0); }
  }

  h1 {
    font-size: 22px;
    font-weight: 600;
    margin: 0;
    letter-spacing: -0.01em;
  }

  .subtitle {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .meta {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-muted);
    text-align: right;
  }

  main {
    padding: 32px 40px 80px;
    max-width: 1200px;
    margin: 0 auto;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 20px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px;
    position: relative;
    overflow: hidden;
    opacity: 0;
    transform: translateY(14px);
    animation: rise 0.5s ease forwards;
    transition: border-color 0.25s ease, transform 0.25s ease, background 0.25s ease;
  }

  .card:hover {
    border-color: var(--brand);
    transform: translateY(-2px);
    background: var(--surface-hover);
  }

  @keyframes rise {
    to { opacity: 1; transform: translateY(0); }
  }

  .card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 18px;
  }

  .asset-name {
    font-size: 19px;
    font-weight: 600;
  }

  .ticker {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-muted);
    letter-spacing: 0.05em;
  }

  .trend-tag {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 4px 10px;
    border-radius: 999px;
    white-space: nowrap;
  }

  .trend-alta   { color: var(--positive); background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.3); }
  .trend-baixa  { color: var(--negative); background: rgba(248, 113, 113, 0.1); border: 1px solid rgba(248, 113, 113, 0.3); }
  .trend-lateral, .trend-null { color: var(--mixed); background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3); }

  .price-row {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 4px;
  }

  .price {
    font-family: var(--font-mono);
    font-size: 30px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  .variation {
    font-family: var(--font-mono);
    font-size: 14px;
    font-weight: 600;
  }

  .variation.positive { color: var(--positive); }
  .variation.negative { color: var(--negative); }

  .divider {
    height: 1px;
    background: var(--border);
    margin: 18px 0;
  }

  .sentiment-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }

  .sentiment-label {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: var(--font-mono);
  }

  .sentiment-value {
    font-weight: 600;
    font-size: 13px;
    text-transform: capitalize;
  }

  .sentiment-value.positivo { color: var(--positive); }
  .sentiment-value.negativo { color: var(--negative); }
  .sentiment-value.misto, .sentiment-value.neutro { color: var(--mixed); }

  .gauge {
    height: 6px;
    border-radius: 999px;
    background: var(--border);
    overflow: hidden;
    margin-bottom: 16px;
  }

  .gauge-fill {
    height: 100%;
    width: 0%;
    border-radius: 999px;
    transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
  }

  .headlines {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .headlines li {
    font-size: 12.5px;
    color: var(--text-muted);
    line-height: 1.5;
    padding-left: 14px;
    position: relative;
  }

  .headlines li::before {
    content: "›";
    position: absolute;
    left: 0;
    color: var(--brand);
  }

  .validation-note {
    margin-top: 16px;
    padding: 10px 12px;
    background: rgba(251, 191, 36, 0.08);
    border: 1px solid rgba(251, 191, 36, 0.25);
    border-radius: 6px;
    font-size: 12px;
    color: var(--mixed);
  }

  .empty-state {
    text-align: center;
    padding: 100px 20px;
    color: var(--text-muted);
  }

  .empty-state .glyph {
    font-family: var(--font-mono);
    font-size: 40px;
    color: var(--border);
    margin-bottom: 16px;
  }

  .empty-state p {
    font-family: var(--font-mono);
    font-size: 13px;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
  }

  .unavailable-badge {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--negative);
    background: rgba(248, 113, 113, 0.1);
    border: 1px solid rgba(248, 113, 113, 0.3);
    padding: 3px 8px;
    border-radius: 6px;
  }

  /* --- Painel de seleccao --- */
  .control-panel {
    display: flex;
    align-items: center;
    gap: 20px;
    flex-wrap: wrap;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 28px;
  }

  .asset-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 999px;
    border: 1px solid var(--border);
    cursor: pointer;
    user-select: none;
    transition: border-color 0.2s ease, background 0.2s ease;
    font-size: 13px;
  }

  .asset-toggle input { accent-color: var(--brand); }

  .asset-toggle.checked {
    border-color: var(--brand);
    background: rgba(94, 234, 212, 0.08);
  }

  .asset-toggle.disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }

  .analyze-btn {
    margin-left: auto;
    background: var(--brand);
    color: #06231F;
    border: none;
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 13.5px;
    padding: 10px 22px;
    border-radius: 8px;
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
  }

  .analyze-btn:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(94, 234, 212, 0.25);
  }

  .analyze-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  /* --- Estados de card: loading / erro --- */
  .card.loading, .card.error-card {
    animation: rise 0.4s ease forwards;
  }

  .skeleton-row {
    height: 14px;
    border-radius: 5px;
    background: linear-gradient(90deg, var(--border) 25%, #2a3b4d 37%, var(--border) 63%);
    background-size: 400% 100%;
    animation: shimmer 1.6s ease infinite;
    margin-bottom: 10px;
  }

  @keyframes shimmer {
    0% { background-position: 100% 50%; }
    100% { background-position: 0 50%; }
  }

  .loading-label {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--brand);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 14px;
  }

  .spinner {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 2px solid rgba(94, 234, 212, 0.25);
    border-top-color: var(--brand);
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .error-card {
    border-color: rgba(248, 113, 113, 0.4);
  }

  .error-message {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--negative);
    line-height: 1.6;
    margin-top: 12px;
  }

  /* --- Pipeline de progresso (dentro do loading card) --- */
  .pipeline {
    display: flex;
    align-items: center;
    margin-bottom: 4px;
  }

  .step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }

  .step-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--border);
    transition: background 0.4s ease, box-shadow 0.4s ease;
  }

  .step-label {
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    white-space: nowrap;
  }

  .step-line {
    flex: 1;
    height: 2px;
    background: var(--border);
    margin: 0 6px 20px;
    transition: background 0.4s ease;
  }

  .step-line.done { background: var(--brand); }

  .step.done .step-dot {
    background: var(--brand);
  }

  .step.done .step-label { color: var(--text); }

  .step.active .step-dot {
    background: var(--brand);
    box-shadow: 0 0 0 4px rgba(94, 234, 212, 0.18);
    animation: dot-pulse 1.4s ease infinite;
  }

  .step.active .step-label { color: var(--brand); }

  @keyframes dot-pulse {
    0%, 100% { box-shadow: 0 0 0 4px rgba(94, 234, 212, 0.18); }
    50% { box-shadow: 0 0 0 8px rgba(94, 234, 212, 0.05); }
  }
</style>
</head>
<body>

<header>
  <div>
    <div class="brand">
      <span class="pulse-dot"></span>
      <h1>Financial Crew</h1>
    </div>
    <div class="subtitle">Análise multi-agente · dados em tempo real</div>
  </div>
  <div class="meta" id="last-updated">a carregar…</div>
</header>

<main>
  <div class="control-panel" id="control-panel">
    <span class="sentiment-label" style="text-transform:none;">A carregar ativos…</span>
  </div>
  <div class="grid" id="grid"></div>
</main>

<script>
const POSITIVE_COLOR = "#34D399";
const NEGATIVE_COLOR = "#F87171";
const MIXED_COLOR = "#FBBF24";

let AVAILABLE_ASSETS = [];
let selectedTickers = new Set();
let pollHandle = null;
const MAX_SELECTION = 2;

function sentimentColor(sentiment) {
  if (sentiment === "positivo") return POSITIVE_COLOR;
  if (sentiment === "negativo") return NEGATIVE_COLOR;
  return MIXED_COLOR;
}

function sentimentGaugeWidth(sentiment) {
  if (sentiment === "positivo") return 85;
  if (sentiment === "negativo") return 15;
  if (sentiment === "misto") return 50;
  return 40;
}

function formatPrice(value, currency) {
  if (value === null || value === undefined) return "—";
  return new Intl.NumberFormat("pt-PT", { style: "currency", currency: currency || "USD" }).format(value);
}

function formatVariation(value) {
  if (value === null || value === undefined) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function renderReportCard(report, index) {
  const tech = report.dados_tecnicos || {};
  const news = report.analise_noticias || {};
  const trend = tech.tendencia || "null";
  const variation = tech.variacao_percentual_1_mes;
  const variationClass = variation > 0 ? "positive" : (variation < 0 ? "negative" : "");
  const sentiment = news.sentimento_geral || "neutro";
  const headlines = (news.titulos_usados || []).slice(0, 4);

  const card = document.createElement("div");
  card.className = "card";
  card.style.animationDelay = `${index * 70}ms`;

  card.innerHTML = `
    <div class="card-top">
      <div>
        <div class="asset-name">${report.ativo}</div>
        <div class="ticker">${report.ticker}</div>
      </div>
      <div class="trend-tag trend-${trend}">${trend === "null" ? "s/ dados" : trend}</div>
    </div>

    <div class="price-row">
      <div class="price">${formatPrice(tech.preco_atual, tech.moeda)}</div>
      <div class="variation ${variationClass}">${formatVariation(variation)}</div>
    </div>
    ${tech.dados_indisponiveis ? '<span class="unavailable-badge">dados técnicos indisponíveis</span>' : ""}

    <div class="divider"></div>

    <div class="sentiment-row">
      <span class="sentiment-label">Sentimento de mercado</span>
      <span class="sentiment-value ${sentiment}">${sentiment}</span>
    </div>
    <div class="gauge">
      <div class="gauge-fill" style="background:${sentimentColor(sentiment)}"></div>
    </div>

    <ul class="headlines">
      ${headlines.map(h => `<li>${h}</li>`).join("")}
    </ul>

    ${report.notas_de_validacao ? `<div class="validation-note">⚠ ${report.notas_de_validacao}</div>` : ""}
  `;

  requestAnimationFrame(() => {
    setTimeout(() => {
      const fill = card.querySelector(".gauge-fill");
      if (fill) fill.style.width = sentimentGaugeWidth(sentiment) + "%";
    }, 150 + index * 70);
  });

  return card;
}

const PIPELINE_STAGES = [
  { key: "noticias", label: "Notícias", detail: "Analista de Notícias a pesquisar manchetes recentes…" },
  { key: "tecnico", label: "Técnico", detail: "Analista Técnico a consultar preço e histórico…" },
  { key: "orquestrador", label: "Orquestrador", detail: "Orquestrador a cruzar e validar os dados…" },
];

function pipelineHTML(stage) {
  if (!stage) {
    const steps = PIPELINE_STAGES.map((s, i) => `
      <div class="step pending">
        <span class="step-dot"></span>
        <span class="step-label">${s.label}</span>
      </div>
      ${i < PIPELINE_STAGES.length - 1 ? '<span class="step-line"></span>' : ""}
    `).join("");
    return `
      <div class="pipeline">${steps}</div>
      <div class="loading-label"><span class="spinner"></span>Em fila de espera…</div>
    `;
  }

  const currentIndex = PIPELINE_STAGES.findIndex(s => s.key === stage);
  const effectiveIndex = stage === "concluido" ? PIPELINE_STAGES.length : currentIndex;
  const activeStage = PIPELINE_STAGES[effectiveIndex] || PIPELINE_STAGES[PIPELINE_STAGES.length - 1];

  const steps = PIPELINE_STAGES.map((s, i) => {
    const state = i < effectiveIndex ? "done" : (i === effectiveIndex ? "active" : "pending");
    return `
      <div class="step ${state}">
        <span class="step-dot"></span>
        <span class="step-label">${s.label}</span>
      </div>
      ${i < PIPELINE_STAGES.length - 1 ? `<span class="step-line ${i < effectiveIndex ? "done" : ""}"></span>` : ""}
    `;
  }).join("");

  return `
    <div class="pipeline">${steps}</div>
    <div class="loading-label"><span class="spinner"></span>${activeStage.detail}</div>
  `;
}

function updatePipelineStage(cardNode, stage) {
  const container = cardNode.querySelector(".pipeline-wrap");
  if (container) container.innerHTML = pipelineHTML(stage);
}

function renderLoadingCard(assetName, ticker, index, stage) {
  const card = document.createElement("div");
  card.className = "card loading";
  card.style.animationDelay = `${index * 70}ms`;
  card.innerHTML = `
    <div class="card-top">
      <div>
        <div class="asset-name">${assetName}</div>
        <div class="ticker">${ticker}</div>
      </div>
    </div>
    <div class="skeleton-row" style="width: 55%; height: 26px;"></div>
    <div class="skeleton-row" style="width: 88%;"></div>
    <div class="divider"></div>
    <div class="pipeline-wrap">${pipelineHTML(stage)}</div>
  `;
  return card;
}

function renderErrorCard(assetName, ticker, index, detail) {
  const card = document.createElement("div");
  card.className = "card error-card";
  card.style.animationDelay = `${index * 70}ms`;
  card.innerHTML = `
    <div class="card-top">
      <div>
        <div class="asset-name">${assetName}</div>
        <div class="ticker">${ticker}</div>
      </div>
      <div class="trend-tag trend-baixa">falhou</div>
    </div>
    <div class="error-message">${detail || "Ocorreu um erro desconhecido."}</div>
  `;
  return card;
}

function renderControlPanel() {
  const panel = document.getElementById("control-panel");
  const limitReached = selectedTickers.size >= MAX_SELECTION;

  panel.innerHTML = `
    <span class="sentiment-label" style="text-transform:none; white-space:nowrap;">
      Escolhe até ${MAX_SELECTION} ativos:
    </span>
  ` + AVAILABLE_ASSETS.map(a => {
    const checked = selectedTickers.has(a.ticker);
    const disabled = !checked && limitReached;
    return `
      <label class="asset-toggle ${checked ? "checked" : ""} ${disabled ? "disabled" : ""}" data-ticker="${a.ticker}">
        <input type="checkbox" ${checked ? "checked" : ""} ${disabled ? "disabled" : ""} />
        ${a.asset_name} <span class="ticker">${a.ticker}</span>
      </label>
    `;
  }).join("") + `<button class="analyze-btn" id="analyze-btn">Analisar selecionados</button>`;

  panel.querySelectorAll(".asset-toggle:not(.disabled)").forEach(el => {
    el.addEventListener("click", (e) => {
      e.preventDefault();
      const ticker = el.dataset.ticker;
      if (selectedTickers.has(ticker)) {
        selectedTickers.delete(ticker);
      } else if (selectedTickers.size < MAX_SELECTION) {
        selectedTickers.add(ticker);
      }
      renderControlPanel();
    });
  });

  document.getElementById("analyze-btn").addEventListener("click", triggerAnalysis);
}

async function triggerAnalysis() {
  if (!selectedTickers.size) return;
  const btn = document.getElementById("analyze-btn");
  btn.disabled = true;
  btn.textContent = "A analisar…";

  await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tickers: Array.from(selectedTickers) }),
  });

  startPolling();
}

let cardElements = {};
let cardSignatures = {};

function orderedTickers(jobs) {
  const known = AVAILABLE_ASSETS.map(a => a.ticker).filter(t => t in jobs);
  const extra = Object.keys(jobs).filter(t => !known.includes(t));
  return known.concat(extra);
}

function renderGrid(jobs) {
  const grid = document.getElementById("grid");
  const tickers = orderedTickers(jobs);

  if (!tickers.length) {
    if (grid.dataset.empty !== "1") {
      grid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <div class="glyph">{ }</div>
          <p>Seleciona um ou mais ativos acima e clica em "Analisar selecionados" para começar.</p>
        </div>`;
      grid.dataset.empty = "1";
      cardElements = {};
      cardSignatures = {};
    }
    return;
  }

  if (grid.dataset.empty === "1") {
    grid.innerHTML = "";
    grid.dataset.empty = "0";
  }

  tickers.forEach((ticker, i) => {
    const job = jobs[ticker];
    const sig = `${job.status}:${job.stage || ""}`;
    const existing = cardElements[ticker];

    // Se ja esta a mostrar "running" e so a fase mudou, atualiza o
    // pipeline no lugar -- sem recriar o cartao, sem reiniciar animacoes,
    // sem flicker. So a barra de progresso do pipeline muda.
    if (job.status === "running" && existing && (cardSignatures[ticker] || "").startsWith("running:")) {
      if (cardSignatures[ticker] !== sig) {
        updatePipelineStage(existing, job.stage);
        cardSignatures[ticker] = sig;
      }
      if (grid.children[i] !== existing) grid.insertBefore(existing, grid.children[i] || null);
      return;
    }

    // Nada mudou desde o ultimo render -- nao tocar no DOM.
    if (cardSignatures[ticker] === sig && existing) {
      if (grid.children[i] !== existing) grid.insertBefore(existing, grid.children[i] || null);
      return;
    }

    let node;
    if (job.status === "success" && job.report) {
      node = renderReportCard(job.report, i);
    } else if (job.status === "error") {
      node = renderErrorCard(job.asset_name || ticker, ticker, i, job.detail);
    } else if (job.status === "running") {
      node = renderLoadingCard(job.asset_name || ticker, ticker, i, job.stage);
    } else {
      node = renderLoadingCard(job.asset_name || ticker, ticker, i, null);
    }

    if (existing && existing.parentNode === grid) {
      grid.replaceChild(node, existing);
    } else {
      grid.insertBefore(node, grid.children[i] || null);
    }
    cardElements[ticker] = node;
    cardSignatures[ticker] = sig;
  });
}

async function fetchStatus() {
  const lastUpdated = document.getElementById("last-updated");
  try {
    const res = await fetch("/api/status");
    const jobs = await res.json();
    renderGrid(jobs);
    lastUpdated.textContent = "atualizado " + new Date().toLocaleTimeString("pt-PT");

    const stillWorking = Object.values(jobs).some(j => j.status === "queued" || j.status === "running");
    const btn = document.getElementById("analyze-btn");
    if (!stillWorking && btn) {
      btn.disabled = false;
      btn.textContent = "Analisar selecionados";
      stopPolling();
    }
  } catch (err) {
    lastUpdated.textContent = "erro ao ligar ao backend";
  }
}

function startPolling() {
  if (pollHandle) return;
  fetchStatus();
  pollHandle = setInterval(fetchStatus, 4000);
}

function stopPolling() {
  clearInterval(pollHandle);
  pollHandle = null;
}

async function init() {
  const res = await fetch("/api/assets");
  AVAILABLE_ASSETS = await res.json();
  selectedTickers = new Set(AVAILABLE_ASSETS.slice(0, MAX_SELECTION).map(a => a.ticker));
  renderControlPanel();
  await fetchStatus();

  const statusRes = await fetch("/api/status");
  const jobs = await statusRes.json();
  const stillWorking = Object.values(jobs).some(j => j.status === "queued" || j.status === "running");
  if (stillWorking) startPolling();
}

init();
</script>

</body>
</html>
"""


if __name__ == "__main__":
    # use_reloader=False: o reloader do Flask reimporta o modulo num processo
    # filho, o que duplicaria a worker thread e o loop de analise. Como o
    # dashboard nao precisa de live-reload de codigo para ser util, desligamos.
    app.run(debug=True, port=5000, use_reloader=False)
