"""
Cache de Resultados de Análise (SQLite + TTL).

Evita voltar a chamar a Groq para o mesmo ativo dentro de uma janela de
tempo curta -- poupa quota (rate limit por minuto e por dia) e torna o
frontend instantâneo quando o utilizador reanalisa o mesmo ativo pouco
depois. SQLite (em vez de um dict em memória) foi escolhido de propósito:
sobrevive a reinícios do processo (o `main.py` e o `frontend/app.py` são
processos separados que partilham a mesma cache em disco).
"""

import json
import logging
import sqlite3
import time
from pathlib import Path

from schemas import RelatorioFinanceiro

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "cache" / "analysis_cache.db"
DEFAULT_TTL_SECONDS = 15 * 60  # 15 minutos


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_cache (
            ticker TEXT PRIMARY KEY,
            asset_name TEXT NOT NULL,
            report_json TEXT NOT NULL,
            cached_at REAL NOT NULL
        )
        """)
    return conn


def get_cached_report(
    ticker: str, ttl_seconds: int = DEFAULT_TTL_SECONDS
) -> RelatorioFinanceiro | None:
    """
    Devolve o relatório em cache para `ticker`, se existir e ainda estiver
    dentro do TTL. Devolve None em qualquer outro caso (cache miss,
    expirado, ou corrompido) -- nunca lança exceção para o chamador, uma
    falha na cache nunca deve impedir a análise real de correr.
    """
    try:
        with _get_connection() as conn:
            row = conn.execute(
                "SELECT report_json, cached_at FROM analysis_cache WHERE ticker = ?",
                (ticker,),
            ).fetchone()
    except sqlite3.Error as exc:
        logger.warning("Erro ao ler cache para %s: %s", ticker, exc)
        return None

    if row is None:
        return None

    report_json, cached_at = row
    age = time.time() - cached_at
    if age > ttl_seconds:
        logger.info(
            "Cache expirado para %s (idade: %.0fs, TTL: %ds) -- vai analisar de novo.",
            ticker,
            age,
            ttl_seconds,
        )
        return None

    try:
        data = json.loads(report_json)
        return RelatorioFinanceiro.model_validate(data)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Cache corrompida para %s, a ignorar: %s", ticker, exc)
        return None


def set_cached_report(
    ticker: str, asset_name: str, report: RelatorioFinanceiro
) -> None:
    """Guarda (ou substitui) o relatório em cache para `ticker`."""
    try:
        with _get_connection() as conn:
            conn.execute(
                """
                INSERT INTO analysis_cache (ticker, asset_name, report_json, cached_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    asset_name = excluded.asset_name,
                    report_json = excluded.report_json,
                    cached_at = excluded.cached_at
                """,
                (
                    ticker,
                    asset_name,
                    json.dumps(report.model_dump(), ensure_ascii=False),
                    time.time(),
                ),
            )
    except sqlite3.Error as exc:
        logger.warning("Erro ao escrever cache para %s: %s", ticker, exc)


def clear_cache() -> None:
    """Remove todas as entradas da cache. Usado principalmente nos testes."""
    try:
        with _get_connection() as conn:
            conn.execute("DELETE FROM analysis_cache")
    except sqlite3.Error as exc:
        logger.warning("Erro ao limpar a cache: %s", exc)


if __name__ == "__main__":
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT ticker, asset_name, cached_at FROM analysis_cache ORDER BY cached_at DESC"
        ).fetchall()

    if not rows:
        print("A cache está vazia.")
    else:
        print(f"{'Ticker':<12} {'Ativo':<12} {'Idade (min)':<12}")
        for ticker, asset_name, cached_at in rows:
            age_min = (time.time() - cached_at) / 60
            print(f"{ticker:<12} {asset_name:<12} {age_min:<12.1f}")
