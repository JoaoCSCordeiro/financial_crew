"""
Ferramenta de Dados Financeiros — Agente 2 (Analista Técnico).

Responsabilidade única: obter dados factuais e determinísticos do Yahoo
Finance via `yfinance`. Esta ferramenta NUNCA interpreta ou opina sobre os
dados — isso é trabalho do LLM a jusante. Aqui garantimos apenas que os
números que chegam ao agente são reais e vêm com proveniência clara.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

import yfinance as yf
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tools.exceptions import InvalidTickerError, MarketDataUnavailableError

logger = logging.getLogger(__name__)

# Erros de rede/infraestrutura que justificam retry automático.
# Um ticker inválido NÃO entra aqui — não faz sentido re-tentar um erro de dados.
_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError)


class MarketDataSchema(BaseModel):
    """Schema de input validado para a ferramenta de dados de mercado."""

    ticker: str = Field(
        ..., description="Símbolo do ativo, ex: 'AAPL', 'TSLA', 'BTC-USD'"
    )


@retry(
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _fetch_ticker_object(ticker: str) -> yf.Ticker:
    """Wrapper de baixo nível com retry exponencial para falhas de rede transitórias."""
    return yf.Ticker(ticker)


def get_market_data(ticker: str) -> Dict[str, Any]:
    """
    Obtém o preço atual e o histórico de 1 mês para um ticker.

    Args:
        ticker: Símbolo do ativo (ex: "AAPL", "TSLA", "BTC-USD", "ETH-USD").

    Returns:
        Dicionário estruturado com preço atual, variação, e resumo do
        histórico de 1 mês (min, max, médias, primeiro/último fecho).

    Raises:
        InvalidTickerError: se o ticker não existir ou não devolver dados.
        MarketDataUnavailableError: se a rede/API falhar após retries.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise InvalidTickerError("O ticker fornecido está vazio.")

    try:
        yf_ticker = _fetch_ticker_object(ticker)
        history = yf_ticker.history(period="1mo", interval="1d")
    except _RETRYABLE_EXCEPTIONS as exc:
        logger.error("Falha de rede ao obter dados para %s: %s", ticker, exc)
        raise MarketDataUnavailableError(
            f"Não foi possível contactar o Yahoo Finance para '{ticker}' após múltiplas tentativas."
        ) from exc
    except (
        Exception
    ) as exc:  # yfinance pode lançar exceções não tipadas em falhas internas
        logger.error("Erro inesperado do yfinance para %s: %s", ticker, exc)
        raise MarketDataUnavailableError(
            f"Erro inesperado ao consultar dados de mercado para '{ticker}': {exc}"
        ) from exc

    if history.empty:
        raise InvalidTickerError(
            f"Ticker '{ticker}' inválido ou sem dados históricos disponíveis. "
            "Verifica se o símbolo está correto (ex: criptomoedas usam sufixo '-USD')."
        )

    try:
        info = yf_ticker.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    except Exception as exc:
        logger.warning(
            "Não foi possível obter 'info' detalhada para %s: %s", ticker, exc
        )
        info = {}
        current_price = None

    last_close = float(history["Close"].iloc[-1])
    first_close = float(history["Close"].iloc[0])
    price = current_price if current_price is not None else last_close

    variation_pct = (
        ((last_close - first_close) / first_close) * 100 if first_close else 0.0
    )

    return {
        "ticker": ticker,
        "data_recolha_utc": datetime.now(timezone.utc).isoformat(),
        "preco_atual": round(float(price), 4),
        "moeda": info.get("currency", "USD"),
        "historico_1_mes": {
            "primeiro_fecho": round(first_close, 4),
            "ultimo_fecho": round(last_close, 4),
            "maximo": round(float(history["High"].max()), 4),
            "minimo": round(float(history["Low"].min()), 4),
            "media_fecho": round(float(history["Close"].mean()), 4),
            "variacao_percentual_1_mes": round(variation_pct, 2),
            "volume_medio": (
                int(history["Volume"].mean()) if "Volume" in history else None
            ),
            "num_sessoes": len(history),
        },
        "nome_empresa": info.get("shortName") or info.get("longName") or ticker,
    }


if __name__ == "__main__":
    # Teste manual rápido do bloco 1 (sem dependência do CrewAI/LLM).
    import json

    for test_ticker in ["AAPL", "TICKER_INVALIDO_XYZ"]:
        print(f"\n--- Teste: {test_ticker} ---")
        try:
            result = get_market_data(test_ticker)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except InvalidTickerError as e:
            print(f"[InvalidTickerError] {e}")
        except MarketDataUnavailableError as e:
            print(f"[MarketDataUnavailableError] {e}")
