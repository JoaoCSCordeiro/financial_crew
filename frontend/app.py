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
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(
    0, str(BASE_DIR)
)  # permite `python frontend/app.py` a partir de qualquer diretoria

from crew import describe_rate_limit_error, is_rate_limit_error  # noqa: E402
from crew import run_analysis  # noqa: E402
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


def _save_report_to_disk(
    asset_name: str, ticker: str, report: RelatorioFinanceiro
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filepath = OUTPUT_DIR / f"{ticker}_{timestamp}.json"
    filepath.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return filepath


def _worker_loop() -> None:
    """
    Worker unico que consome a fila de pedidos de analise, um de cada vez.
    Corre para sempre numa thread em background (daemon), iniciada quando
    o modulo e importado.
    """
    while True:
        asset_name, ticker = _job_queue.get()
        _set_job(
            ticker,
            status="running",
            asset_name=asset_name,
            detail=None,
            stage="noticias",
        )

        def _on_stage_change(stage: str, _ticker=ticker) -> None:
            _set_job(_ticker, stage=stage)

        try:
            result = run_analysis(
                asset_name=asset_name, ticker=ticker, stage_callback=_on_stage_change
            )
        except LLMConfigError as exc:
            _set_job(
                ticker, status="error", detail=f"Erro de configuração do LLM: {exc}"
            )
        except FinancialToolError as exc:
            _set_job(
                ticker, status="error", detail=f"Erro numa ferramenta de dados: {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            if is_rate_limit_error(exc):
                _set_job(
                    ticker,
                    status="error",
                    detail=describe_rate_limit_error(exc),
                )
            else:
                _set_job(
                    ticker,
                    status="error",
                    detail=f"Erro inesperado: {type(exc).__name__}: {exc}",
                )
        else:
            if isinstance(result, RelatorioFinanceiro):
                _save_report_to_disk(asset_name, ticker, result)
                _set_job(
                    ticker, status="success", report=result.model_dump(), detail=None
                )
            else:
                _set_job(
                    ticker,
                    status="error",
                    detail="O LLM não devolveu um relatório JSON válido.",
                )
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
        _set_job(
            ticker, status="success", asset_name=asset_name, report=data, detail=None
        )


_load_existing_reports()

_worker_thread = threading.Thread(target=_worker_loop, daemon=True)
_worker_thread.start()


@app.route("/api/assets")
def api_assets():
    """Devolve os ativos disponiveis para o frontend construir a seleccao."""
    return jsonify(
        [
            {"asset_name": name, "ticker": ticker}
            for name, ticker in AVAILABLE_ASSETS.items()
        ]
    )


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


@app.route("/api/metrics")
def api_metrics():
    """Devolve o resumo agregado de metricas (ver metrics.py)."""
    from metrics import summarize

    return jsonify(summarize())


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


# ---------------------------------------------------------------------------
# Dashboard -- direção visual "Dossiê de Caso": cada ativo é um processo com
# dois agentes-testemunha (notícias, técnico) e um veredito carimbado pelo
# Orquestrador. Ver DECISOES-FINETUNING.md / histórico da conversa para o
# raciocínio de design -- não é decoração, mapeia a arquitetura real do
# sistema (veto do Agente 3 sobre dados sem proveniência).
# ---------------------------------------------------------------------------
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="pt-PT">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Financial Crew — Dossiê de Análise</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Special+Elite&family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --desk: #1B1712;
    --desk-line: rgba(239, 230, 204, 0.10);
    --desk-ink: #EFE6CC;
    --desk-ink-muted: rgba(239, 230, 204, 0.5);
    --paper: #EFE6CC;
    --paper-edge: #DBCDA2;
    --ink: #241F16;
    --ink-faded: #8A7C5C;
    --stamp-green: #35563A;
    --stamp-red: #A1301F;
    --font-display: 'Special Elite', 'Courier New', monospace;
    --font-mono: 'IBM Plex Mono', monospace;
    --font-sans: 'Inter', -apple-system, 'Segoe UI', sans-serif;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    background: var(--desk);
    background-image:
      repeating-linear-gradient(180deg, rgba(0,0,0,0) 0, rgba(0,0,0,0) 39px, var(--desk-line) 40px),
      radial-gradient(ellipse at 50% -10%, rgba(239,230,204,0.05), transparent 55%);
    color: var(--desk-ink);
    font-family: var(--font-sans);
    min-height: 100vh;
  }

  /* Dados "crus" -- precos, percentagens, tokens, timestamps -- ficam a
     monoespacada; tudo o resto (titulos, listas, menus) usa a sans-serif
     para criar contraste claro entre dado e prosa/navegacao. */

  a, button, input { font-family: inherit; }

  :focus-visible {
    outline: 2px solid var(--stamp-red);
    outline-offset: 2px;
  }

  header {
    padding: 40px 40px 22px;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
    border-bottom: 1px solid var(--desk-line);
  }

  h1 {
    font-family: var(--font-display);
    font-size: 24px;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.01em;
  }

  .subtitle {
    font-size: 11px;
    color: var(--desk-ink-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
  }

  .meta {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--desk-ink-muted);
    text-align: right;
  }

  main {
    padding: 32px 40px 90px;
    max-width: 1200px;
    margin: 0 auto;
  }

  /* Flex em vez de grid auto-fill: uma ultima linha incompleta (ex: 4º
     cartao sozinho) fica centrada no seu tamanho natural em vez de esticar
     para preencher a linha toda. */
  .grid {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 26px;
  }

  /* --- Painel de seleccao (fica na secretaria, nao no papel) --- */
  .control-panel {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
    padding: 16px 0 26px;
    margin-bottom: 8px;
    border-bottom: 1px dashed var(--desk-line);
  }

  .control-label {
    font-size: 11px;
    color: var(--desk-ink-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    white-space: nowrap;
  }

  .asset-chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 8px 16px;
    border-radius: 999px;
    border: 1px solid var(--desk-line);
    background: transparent;
    cursor: pointer;
    user-select: none;
    font-family: var(--font-sans);
    font-size: 12.5px;
    color: var(--desk-ink);
    transition: border-color 0.15s ease, background 0.15s ease, color 0.15s ease;
  }

  .asset-chip .ticker { font-family: var(--font-mono); color: var(--desk-ink-muted); font-size: 11px; }

  .asset-chip:hover:not(:disabled) { border-color: var(--paper-edge); }

  .asset-chip.checked {
    border-color: var(--desk-ink);
    background: var(--desk-ink);
    color: var(--desk);
  }

  .asset-chip.checked .ticker { color: var(--desk); opacity: 0.6; }

  .asset-chip:disabled { opacity: 0.35; cursor: not-allowed; }

  .stamp-btn {
    margin-left: auto;
    background: transparent;
    color: var(--desk-ink);
    border: 1px solid var(--desk-ink);
    font-family: var(--font-sans);
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 10px 20px;
    cursor: pointer;
    transition: transform 0.1s ease, background 0.15s ease, color 0.15s ease;
  }

  .stamp-btn:hover:not(:disabled) {
    background: var(--desk-ink);
    color: var(--desk);
  }

  .stamp-btn:active:not(:disabled) { transform: scale(0.96); }

  .stamp-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* --- Cada processo: uma folha de papel sobre a secretaria --- */
  .card {
    flex: 1 1 320px;
    max-width: 380px;
    background: var(--paper);
    color: var(--ink);
    border: 1px solid rgba(0,0,0,0.15);
    padding: 22px 22px 20px;
    position: relative;
    box-shadow: 0 6px 16px rgba(0,0,0,0.35);
    opacity: 0;
    transform: translateY(10px) rotate(var(--tilt, 0deg));
    animation: settle 0.4s ease forwards;
  }

  .card:nth-child(odd)  { --tilt: -0.6deg; }
  .card:nth-child(even) { --tilt: 0.5deg; }

  .card:hover {
    transform: translateY(-3px) rotate(var(--tilt, 0deg));
    box-shadow: 0 12px 24px rgba(0,0,0,0.4);
  }

  .card, .card:hover { transition: transform 0.2s ease, box-shadow 0.2s ease; }

  @keyframes settle {
    to { opacity: 1; transform: translateY(0) rotate(var(--tilt, 0deg)); }
  }

  .card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 14px;
  }

  .case-no {
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--ink-faded);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .asset-name {
    font-family: var(--font-sans);
    font-weight: 600;
    font-size: 20px;
    margin-top: 3px;
  }

  .trend-tag {
    font-size: 10.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 4px 8px;
    border: 1px solid currentColor;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .trend-alta   { color: var(--stamp-green); }
  .trend-baixa  { color: var(--stamp-red); }
  .trend-lateral, .trend-null { color: var(--ink-faded); }

  .rule {
    height: 1px;
    background: var(--paper-edge);
    margin: 16px 0;
  }

  .eyebrow {
    font-size: 10px;
    color: var(--ink-faded);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 9px;
  }

  .price-row {
    display: flex;
    align-items: baseline;
    gap: 12px;
  }

  .price {
    font-family: var(--font-mono);
    font-size: 26px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  .variation { font-family: var(--font-mono); font-size: 13px; font-weight: 600; }
  .variation.positive { color: var(--stamp-green); }
  .variation.negative { color: var(--stamp-red); }

  /* Badge de sentimento -- pilula, nao texto entre aspas. */
  .sentiment-value {
    display: inline-block;
    font-family: var(--font-sans);
    font-weight: 600;
    font-size: 12px;
    text-transform: capitalize;
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 10px;
  }

  .sentiment-value.positivo {
    background: #C9E8CE;
    border: 1px solid rgba(53, 86, 58, 0.25);
    color: #1F4027;
  }

  .sentiment-value.negativo {
    background: #F3D0C8;
    border: 1px solid rgba(161, 48, 31, 0.25);
    color: #7A2016;
  }

  .sentiment-value.misto, .sentiment-value.neutro {
    background: rgba(138, 124, 92, 0.15);
    border: 1px solid rgba(138, 124, 92, 0.3);
    color: var(--ink-faded);
  }

  .headlines {
    margin: 0;
    padding-left: 20px;
    display: flex;
    flex-direction: column;
  }

  .headlines li {
    font-size: 12px;
    color: var(--ink-faded);
    line-height: 1.6;
    margin-bottom: 10px;
  }

  .headlines li:last-child { margin-bottom: 0; }

  .unavailable-badge {
    display: inline-block;
    font-size: 10.5px;
    color: var(--stamp-red);
    border: 1px solid var(--stamp-red);
    padding: 2px 7px;
    margin-top: 8px;
  }

  /* --- O carimbo de veredito: a assinatura da pagina --- */
  .verdict-wrap {
    display: flex;
    justify-content: flex-end;
    margin-top: 18px;
  }

  .stamp {
    font-family: var(--font-display);
    font-size: 13px;
    letter-spacing: 0.12em;
    padding: 7px 14px;
    border: 2px solid currentColor;
    border-radius: 2px;
    text-shadow: 0.5px 0.5px 0 currentColor;
    opacity: 0.85;
    transform: rotate(var(--r, -2deg));
    animation: stamp-down 0.35s cubic-bezier(0.2, 0.9, 0.3, 1.2) both;
  }

  .stamp-green { color: var(--stamp-green); }
  .stamp-red   { color: var(--stamp-red); }

  @keyframes stamp-down {
    0%   { opacity: 0; transform: scale(1.9) rotate(var(--r, -2deg)); }
    60%  { opacity: 0.92; transform: scale(0.94) rotate(var(--r, -2deg)); }
    100% { opacity: 0.85; transform: scale(1) rotate(var(--r, -2deg)); }
  }

  .validation-note {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px dashed var(--paper-edge);
    font-size: 11.5px;
    color: var(--stamp-red);
    line-height: 1.5;
  }

  .validation-note .eyebrow { color: var(--stamp-red); opacity: 0.8; }

  /* --- Estado vazio --- */
  .empty-state {
    text-align: center;
    padding: 90px 20px;
    color: var(--desk-ink-muted);
    border: 1px dashed var(--desk-line);
  }

  .empty-state .glyph {
    font-family: var(--font-display);
    font-size: 15px;
    color: var(--desk-ink-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 14px;
  }

  .empty-state p {
    font-size: 13px;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.7;
  }

  /* --- Card em curso / erro: continuam folhas, ainda sem carimbo --- */
  .card.loading .asset-name { color: var(--ink-faded); }

  .case-log {
    display: flex;
    flex-direction: column;
    gap: 9px;
    font-size: 12px;
  }

  .case-log .row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    color: var(--ink-faded);
  }

  .case-log .row.done { color: var(--ink); }
  .case-log .row.active { color: var(--ink); }

  .case-log .mark { flex-shrink: 0; width: 16px; text-align: center; }
  .case-log .row.done .mark { color: var(--stamp-green); }

  .cursor {
    display: inline-block;
    width: 7px;
    height: 12px;
    background: var(--ink);
    margin-left: 2px;
    animation: blink 1s step-end infinite;
    vertical-align: -2px;
  }

  @keyframes blink { 50% { opacity: 0; } }

  .error-card .asset-name { color: var(--stamp-red); }

  .error-message {
    font-size: 12px;
    color: var(--ink);
    line-height: 1.6;
  }

  /* --- Registo (metricas) --- */
  .ledger {
    display: flex;
    gap: 30px;
    flex-wrap: wrap;
    padding: 12px 40px;
    border-bottom: 1px solid var(--desk-line);
    font-size: 11px;
  }

  .ledger-item { display: flex; flex-direction: column; gap: 2px; }

  .ledger-value {
    font-family: var(--font-mono);
    font-size: 16px;
    font-weight: 600;
    color: var(--desk-ink);
    font-variant-numeric: tabular-nums;
  }

  .ledger-label {
    color: var(--desk-ink-muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
  }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.001ms !important; animation-iteration-count: 1 !important; transition-duration: 0.001ms !important; }
  }

  @media (max-width: 640px) {
    header, main, .ledger { padding-left: 20px; padding-right: 20px; }
    .stamp-btn { margin-left: 0; width: 100%; }
  }
</style>
</head>
<body>

<header>
  <div>
    <h1>Financial Crew</h1>
    <div class="subtitle">Dossiê de análise · três agentes, um veredito</div>
  </div>
  <div class="meta" id="last-updated">a carregar…</div>
</header>

<div class="ledger" id="ledger"></div>

<main>
  <div class="control-panel" id="control-panel">
    <span class="control-label">a carregar ativos…</span>
  </div>
  <div class="grid" id="grid"></div>
</main>

<script>
let AVAILABLE_ASSETS = [];
let selectedTickers = new Set();
let pollHandle = null;
const MAX_SELECTION = 2;

function trendLabel(trend) {
  if (trend === "alta") return "ALTA ▲";
  if (trend === "baixa") return "BAIXA ▼";
  if (trend === "lateral") return "LATERAL ▬";
  return "S/ DADOS";
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

// Angulo do carimbo derivado do id do ativo (nao aleatorio a cada render --
// o mesmo cartao fica sempre com o mesmo angulo). Intervalo pequeno,
// -3deg a 3deg, para parecer autentico sem ser descuidado.
function stampAngle(seed) {
  let hash = 0;
  for (let i = 0; i < seed.length; i++) {
    hash = (hash * 31 + seed.charCodeAt(i)) | 0;
  }
  return (Math.abs(hash) % 7) - 3;
}

function renderReportCard(report, index) {
  const tech = report.dados_tecnicos || {};
  const news = report.analise_noticias || {};
  const trend = tech.tendencia || "null";
  const variation = tech.variacao_percentual_1_mes;
  const variationClass = variation > 0 ? "positive" : (variation < 0 ? "negative" : "");
  const sentiment = news.sentimento_geral || "neutro";
  const headlines = (news.titulos_usados || []).slice(0, 4);
  const flagged = Boolean(report.notas_de_validacao);
  const rot = stampAngle(report.ticker);

  const card = document.createElement("div");
  card.className = "card";
  card.style.animationDelay = `${index * 60}ms`;

  card.innerHTML = `
    <div class="card-top">
      <div>
        <div class="case-no">Nº ${report.ticker}</div>
        <div class="asset-name">${report.ativo}</div>
      </div>
      <div class="trend-tag trend-${trend}">${trendLabel(trend)}</div>
    </div>

    <div class="eyebrow">Testemunho — analista técnico</div>
    <div class="price-row">
      <div class="price">${formatPrice(tech.preco_atual, tech.moeda)}</div>
      <div class="variation ${variationClass}">${formatVariation(variation)}</div>
    </div>
    ${tech.dados_indisponiveis ? '<span class="unavailable-badge">dados indisponíveis</span>' : ""}

    <div class="rule"></div>

    <div class="eyebrow">Testemunho — analista de notícias</div>
    <div class="sentiment-value ${sentiment}">${sentiment}</div>
    <ol class="headlines">
      ${headlines.map(h => `<li>${h}</li>`).join("")}
    </ol>

    <div class="verdict-wrap">
      <div class="stamp ${flagged ? "stamp-red" : "stamp-green"}" style="--r:${rot}deg">
        ${flagged ? "Sinalizado" : "Validado"}
      </div>
    </div>
    ${flagged ? `
      <div class="validation-note">
        <div class="eyebrow">Nota do orquestrador</div>
        ${report.notas_de_validacao}
      </div>` : ""}
  `;

  return card;
}

const PIPELINE_STAGES = [
  { key: "noticias", label: "Notícias recebidas" },
  { key: "tecnico", label: "Dados técnicos" },
  { key: "orquestrador", label: "Validação final" },
];

function pipelineHTML(stage) {
  const currentIndex = stage ? PIPELINE_STAGES.findIndex(s => s.key === stage) : -1;
  const effectiveIndex = stage === "concluido" ? PIPELINE_STAGES.length : currentIndex;

  const rows = PIPELINE_STAGES.map((s, i) => {
    const state = i < effectiveIndex ? "done" : (i === effectiveIndex ? "active" : "pending");
    const mark = state === "done" ? "✓" : (state === "active" ? "…" : "·");
    return `
      <div class="row ${state}">
        <span>${i + 1}. ${s.label}${state === "active" ? '<span class="cursor"></span>' : ""}</span>
        <span class="mark">${mark}</span>
      </div>`;
  }).join("");

  return `<div class="case-log">${rows}</div>`;
}

function updatePipelineStage(cardNode, stage) {
  const container = cardNode.querySelector(".pipeline-wrap");
  if (container) container.innerHTML = pipelineHTML(stage);
}

function renderLoadingCard(assetName, ticker, index, stage) {
  const card = document.createElement("div");
  card.className = "card loading";
  card.style.animationDelay = `${index * 60}ms`;
  card.innerHTML = `
    <div class="card-top">
      <div>
        <div class="case-no">Nº ${ticker}</div>
        <div class="asset-name">${assetName}</div>
      </div>
    </div>
    <div class="rule"></div>
    <div class="eyebrow">Registo do processo</div>
    <div class="pipeline-wrap">${pipelineHTML(stage)}</div>
  `;
  return card;
}

function renderErrorCard(assetName, ticker, index, detail) {
  const card = document.createElement("div");
  card.className = "card error-card";
  card.style.animationDelay = `${index * 60}ms`;
  card.innerHTML = `
    <div class="card-top">
      <div>
        <div class="case-no">Nº ${ticker}</div>
        <div class="asset-name">${assetName}</div>
      </div>
      <div class="trend-tag trend-baixa">Rejeitado</div>
    </div>
    <div class="rule"></div>
    <div class="eyebrow">Notas do processo</div>
    <div class="error-message">${detail || "Ocorreu um erro desconhecido."}</div>
  `;
  return card;
}

function renderControlPanel() {
  const panel = document.getElementById("control-panel");
  const limitReached = selectedTickers.size >= MAX_SELECTION;

  panel.innerHTML = `
    <span class="control-label">Escolhe até ${MAX_SELECTION} ativos para abrir processo:</span>
  ` + AVAILABLE_ASSETS.map(a => {
    const checked = selectedTickers.has(a.ticker);
    const disabled = !checked && limitReached;
    return `
      <button type="button" class="asset-chip ${checked ? "checked" : ""}" data-ticker="${a.ticker}" aria-pressed="${checked}" ${disabled ? "disabled" : ""}>
        ${a.asset_name} <span class="ticker">${a.ticker}</span>
      </button>
    `;
  }).join("") + `<button class="stamp-btn" id="analyze-btn">Abrir processo</button>`;

  panel.querySelectorAll(".asset-chip:not(:disabled)").forEach(el => {
    el.addEventListener("click", () => {
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
  btn.textContent = "A investigar…";

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
        <div class="empty-state" style="flex-basis: 100%;">
          <div class="glyph">Nenhum processo aberto</div>
          <p>Seleciona um ou mais ativos acima e carrega em "Abrir processo" para começar.</p>
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
    // registo do processo no lugar -- sem recriar o cartao, sem reiniciar
    // animacoes, sem flicker.
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
      btn.textContent = "Abrir processo";
      stopPolling();
      loadMetrics();
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

async function loadMetrics() {
  const bar = document.getElementById("ledger");
  try {
    const res = await fetch("/api/metrics");
    const m = await res.json();

    if (!m.total_runs) {
      bar.innerHTML = `<span class="ledger-label">Registo vazio -- corre uma análise para veres métricas aqui.</span>`;
      return;
    }

    const successPct = m.success_rate !== null ? Math.round(m.success_rate * 100) : "—";
    bar.innerHTML = `
      <div class="ledger-item"><span class="ledger-value">${m.total_runs}</span><span class="ledger-label">Processos</span></div>
      <div class="ledger-item"><span class="ledger-value">${successPct}%</span><span class="ledger-label">Taxa de sucesso</span></div>
      <div class="ledger-item"><span class="ledger-value">${m.avg_duration_seconds ?? "—"}s</span><span class="ledger-label">Duração média</span></div>
      <div class="ledger-item"><span class="ledger-value">${(m.total_tokens || 0).toLocaleString("pt-PT")}</span><span class="ledger-label">Tokens (total)</span></div>
    `;
  } catch (err) {
    bar.innerHTML = "";
  }
}

async function init() {
  const res = await fetch("/api/assets");
  AVAILABLE_ASSETS = await res.json();
  selectedTickers = new Set(AVAILABLE_ASSETS.slice(0, MAX_SELECTION).map(a => a.ticker));
  renderControlPanel();
  await fetchStatus();
  await loadMetrics();

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
    # host="0.0.0.0": necessário para o servidor ser acessível de fora do
    # contentor Docker (o default do Flask, 127.0.0.1, só aceita ligações
    # da própria máquina). FLASK_DEBUG=0 no .env/docker-compose desliga o
    # modo debug em produção.
    debug_mode = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode, use_reloader=False)
