"""
Metricas e Observabilidade.

Cada execucao de `run_analysis` (sucesso ou falha) e registada como uma
linha JSON em `metrics/metrics.jsonl` -- formato JSON Lines para poder
crescer indefinidamente sem reescrever o ficheiro todo a cada execucao, e
para ser trivialmente processavel linha a linha (ex: por `jq`, pandas, ou
o `summarize()` abaixo).

O que e registado por execucao:
- asset_name, ticker
- status: "success" | "invalid_schema" | "error"
- duration_seconds: tempo total do kickoff (inclui as pausas propositadas
  entre tarefas -- ver tasks.py -- por isso nao e "tempo de LLM puro", e
  "tempo de parede" da analise completa)
- tokens: total/prompt/completion/successful_requests, quando disponivel
  (crew_output.token_usage do CrewAI)
- error: mensagem de erro, quando status != "success"
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

METRICS_DIR = Path(__file__).resolve().parent / "metrics"
METRICS_FILE = METRICS_DIR / "metrics.jsonl"


def record_analysis(
    asset_name: str,
    ticker: str,
    duration_seconds: float,
    status: str,
    tokens: dict[str, int] | None = None,
    error: str | None = None,
) -> None:
    """
    Regista uma execucao de analise no ficheiro de metricas.

    Nunca lanca excecao para o chamador -- uma falha ao escrever metricas
    (ex: disco cheio, permissoes) nao deve derrubar a analise em si. Em
    caso de falha, regista um aviso no logger e segue em frente.
    """
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "asset_name": asset_name,
        "ticker": ticker,
        "status": status,
        "duration_seconds": round(duration_seconds, 2),
        "tokens": tokens,
        "error": error,
    }

    try:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with METRICS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Não foi possível escrever métricas em disco: %s", exc)


def read_metrics() -> list[dict[str, Any]]:
    """Le todas as entradas de metricas registadas ate agora."""
    if not METRICS_FILE.exists():
        return []

    entries = []
    with METRICS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def summarize() -> dict[str, Any]:
    """
    Agrega as metricas registadas num resumo de alto nivel: total de
    execucoes, taxa de sucesso, duracao media, tokens totais consumidos,
    e a execucao mais lenta/mais cara em tokens (util para identificar
    onde otimizar).
    """
    entries = read_metrics()
    if not entries:
        return {
            "total_runs": 0,
            "success_rate": None,
            "avg_duration_seconds": None,
            "total_tokens": 0,
            "runs_by_status": {},
            "slowest_run": None,
            "most_token_intensive_run": None,
        }

    total_runs = len(entries)
    successes = [e for e in entries if e["status"] == "success"]
    runs_by_status: dict[str, int] = {}
    for e in entries:
        runs_by_status[e["status"]] = runs_by_status.get(e["status"], 0) + 1

    durations = [
        e["duration_seconds"] for e in entries if e.get("duration_seconds") is not None
    ]
    avg_duration = round(sum(durations) / len(durations), 2) if durations else None

    total_tokens = sum((e.get("tokens") or {}).get("total", 0) or 0 for e in entries)

    slowest_run = max(
        entries, key=lambda e: e.get("duration_seconds") or 0, default=None
    )
    most_token_intensive_run = max(
        entries,
        key=lambda e: (e.get("tokens") or {}).get("total", 0) or 0,
        default=None,
    )

    return {
        "total_runs": total_runs,
        "success_rate": round(len(successes) / total_runs, 3) if total_runs else None,
        "avg_duration_seconds": avg_duration,
        "total_tokens": total_tokens,
        "runs_by_status": runs_by_status,
        "slowest_run": slowest_run,
        "most_token_intensive_run": most_token_intensive_run,
    }


if __name__ == "__main__":
    summary = summarize()
    print("--- Resumo de Métricas ---")
    print(f"Total de execuções:      {summary['total_runs']}")
    print(f"Taxa de sucesso:         {summary['success_rate']}")
    print(f"Duração média (s):       {summary['avg_duration_seconds']}")
    print(f"Tokens totais usados:    {summary['total_tokens']}")
    print(f"Execuções por status:    {summary['runs_by_status']}")
    if summary["slowest_run"]:
        r = summary["slowest_run"]
        print(
            f"Execução mais lenta:     {r['asset_name']} ({r['ticker']}) -- {r['duration_seconds']}s"
        )
    if summary["most_token_intensive_run"]:
        r = summary["most_token_intensive_run"]
        tokens = (r.get("tokens") or {}).get("total", 0)
        print(
            f"Execução mais cara:      {r['asset_name']} ({r['ticker']}) -- {tokens} tokens"
        )
