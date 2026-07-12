"""
Entrypoint de Producao - Bloco 5.

Envolve `run_analysis()` (crew.py) numa funcao `main` robusta:
  - aceita um ou mais ativos via linha de comando (ou usa a lista default
    do projeto: Apple, Tesla, Bitcoin, Ethereum, Microsoft)
  - trata falhas por ativo de forma isolada (um ativo falhar nao aborta
    os restantes)
  - guarda cada relatorio validado (schemas.RelatorioFinanceiro) como JSON
    em disco, pronto a ser consumido por um frontend ou base de dados
  - devolve um resumo de sucesso/falha no final
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from crew import is_rate_limit_error, run_analysis
from llm_config import LLMConfigError
from schemas import RelatorioFinanceiro
from tools.exceptions import FinancialToolError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Nota: a pausa entre ativos deixou de ser um `time.sleep` fixo aqui -- o
# rate limit da Groq (8000 tokens/minuto) e agora gerido de forma adaptativa
# pelo token bucket em rate_limiter.py, chamado internamente por
# crew.run_analysis. Ver README, "Rate limiting" para detalhes.

# Ativos suportados por default. Reduzido a 2 para caber confortavelmente no
# free tier da Groq (8000 TPM / 200000 TPD) sem esgotar a quota diaria a meio
# de uma execucao. Usa --ticker para analisar outros ativos (Bitcoin, Ethereum,
# Microsoft, etc.) individualmente quando tiveres quota disponivel.
DEFAULT_ASSETS: dict[str, str] = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
}

OUTPUT_DIR = Path("output")


def _save_report(asset_name: str, ticker: str, report: RelatorioFinanceiro) -> Path:
    """Guarda o relatorio validado como JSON em disco e devolve o caminho."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_ticker = ticker.replace("/", "-")
    filepath = OUTPUT_DIR / f"{safe_ticker}_{timestamp}.json"

    filepath.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return filepath


def analyze_one(asset_name: str, ticker: str) -> dict:
    """
    Executa a analise para um unico ativo, com tratamento de erros completo.

    Returns:
        Dicionario de resumo: {"asset_name", "ticker", "status", "detail"}.
        `status` e um de: "success", "invalid_schema", "llm_error", "tool_error", "unexpected_error".
    """
    logger.info("A analisar %s (%s)...", asset_name, ticker)

    try:
        result = run_analysis(asset_name=asset_name, ticker=ticker)
    except LLMConfigError as exc:
        logger.error(
            "Falha de configuracao/ligacao ao LLM para %s: %s", asset_name, exc
        )
        return {
            "asset_name": asset_name,
            "ticker": ticker,
            "status": "llm_error",
            "detail": str(exc),
        }
    except FinancialToolError as exc:
        logger.error("Falha numa ferramenta de dados para %s: %s", asset_name, exc)
        return {
            "asset_name": asset_name,
            "ticker": ticker,
            "status": "tool_error",
            "detail": str(exc),
        }
    except Exception as exc:
        if is_rate_limit_error(exc):
            logger.error(
                "Rate limit da Groq excedido para %s mesmo apos retries com backoff: %s",
                asset_name,
                exc,
            )
            return {
                "asset_name": asset_name,
                "ticker": ticker,
                "status": "rate_limit_error",
                "detail": str(exc),
            }
        # Ultima linha de defesa -- nunca deixamos uma excecao nao tratada
        # abortar o processamento dos restantes ativos num batch.
        logger.exception("Erro inesperado ao analisar %s", asset_name)
        return {
            "asset_name": asset_name,
            "ticker": ticker,
            "status": "unexpected_error",
            "detail": f"{type(exc).__name__}: {exc}",
        }

    if not isinstance(result, RelatorioFinanceiro):
        logger.warning(
            "Não foi possível extrair um RelatorioFinanceiro válido do output "
            "do Orquestrador para %s. Resultado em bruto não será guardado em disco.",
            asset_name,
        )
        return {
            "asset_name": asset_name,
            "ticker": ticker,
            "status": "invalid_schema",
            "detail": "O output do LLM não continha JSON válido que respeitasse o schema.",
        }

    filepath = _save_report(asset_name, ticker, result)
    logger.info("Relatorio de %s guardado em %s", asset_name, filepath)
    return {
        "asset_name": asset_name,
        "ticker": ticker,
        "status": "success",
        "detail": str(filepath),
    }


def main(assets: dict[str, str]) -> int:
    """
    Executa a analise para todos os ativos fornecidos e imprime um resumo.

    Returns:
        Codigo de saida do processo: 0 se todos os ativos tiveram sucesso,
        1 se pelo menos um falhou (util para deteccao em pipelines CI/CD).
    """
    summaries = [
        analyze_one(asset_name, ticker) for asset_name, ticker in assets.items()
    ]

    print("\n" + "=" * 60)
    print("RESUMO DA EXECUCAO")
    print("=" * 60)
    for summary in summaries:
        status_label = summary["status"].upper()
        print(
            f"[{status_label:15}] {summary['asset_name']:12} ({summary['ticker']:8}) -> {summary['detail']}"
        )
    print("=" * 60)

    failures = [s for s in summaries if s["status"] != "success"]
    if failures:
        print(f"\n{len(failures)} de {len(summaries)} ativo(s) falharam.")
        return 1

    print(f"\nTodos os {len(summaries)} ativo(s) foram analisados com sucesso.")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sistema de Analise Financeira Multi-Agente -- execucao de producao."
    )
    parser.add_argument(
        "--ticker",
        action="append",
        metavar="NOME:TICKER",
        help=(
            "Analisar um ativo especifico, no formato 'NomeDoAtivo:TICKER' "
            "(ex: --ticker Tesla:TSLA). Pode ser repetido para varios ativos. "
            "Se omitido, usa a lista default do projeto (Apple, Tesla)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.ticker:
        try:
            selected_assets = dict(pair.split(":", 1) for pair in args.ticker)
        except ValueError:
            print(
                "Erro: --ticker tem de estar no formato 'NomeDoAtivo:TICKER' (ex: Tesla:TSLA)"
            )
            sys.exit(2)
    else:
        selected_assets = DEFAULT_ASSETS

    exit_code = main(selected_assets)
    sys.exit(exit_code)
