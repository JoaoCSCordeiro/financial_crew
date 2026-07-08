"""
Ferramenta de Notícias — Agente 1 (Analista de Notícias de Mercado).

Responsabilidade única: recolher títulos de notícias recentes e factuais
sobre um ativo, via DuckDuckGo (sem chave de API, custo zero). Esta
ferramenta NÃO faz análise de sentimento — apenas devolve os títulos e
fontes em bruto, para que o LLM (Agente 1) faça a interpretação e o
Orquestrador (Agente 3) possa verificar que nenhum título foi inventado.
"""

import logging
from typing import Any, Dict, List

from ddgs import DDGS
from ddgs.exceptions import (
    DDGSException as DuckDuckGoSearchException,
    RatelimitException,
    TimeoutException,
)
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tools.exceptions import NewsSearchError

logger = logging.getLogger(__name__)


class NewsSearchSchema(BaseModel):
    """Schema de input validado para a ferramenta de notícias."""

    asset_name: str = Field(
        ..., description="Nome do ativo em linguagem natural, ex: 'Tesla', 'Bitcoin', 'Apple'"
    )
    max_results: int = Field(default=5, ge=1, le=10, description="Número máximo de títulos a devolver")


@retry(
    retry=retry_if_exception_type((RatelimitException, TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=3, max=20),
    reraise=True,
)
def _run_ddg_news_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Wrapper de baixo nível com retry para rate limits transitórios do DuckDuckGo."""
    with DDGS() as ddgs:
        results = list(ddgs.news(query, max_results=max_results))
    return results


def get_recent_news(asset_name: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Obtém os títulos de notícias mais recentes sobre um ativo financeiro.

    Args:
        asset_name: Nome do ativo em linguagem natural (ex: "Tesla", "Bitcoin").
        max_results: Número máximo de notícias a devolver (default: 5).

    Returns:
        Dicionário com a lista de notícias (título, fonte, data, url, excerto)
        e metadados da pesquisa.

    Raises:
        NewsSearchError: se a pesquisa falhar (rate limit, rede, ou termo vazio).
    """
    asset_name = asset_name.strip()
    if not asset_name:
        raise NewsSearchError("O nome do ativo fornecido está vazio.")

    query = f"{asset_name} stock news"

    try:
        raw_results = _run_ddg_news_search(query, max_results)
    except (RatelimitException, TimeoutException) as exc:
        logger.error("Rate limit / timeout do DuckDuckGo para '%s': %s", asset_name, exc)
        raise NewsSearchError(
            f"O DuckDuckGo aplicou rate limiting à pesquisa de '{asset_name}' após múltiplas tentativas. "
            "Tenta novamente dentro de alguns minutos."
        ) from exc
    except DuckDuckGoSearchException as exc:
        logger.error("Erro genérico do DuckDuckGo para '%s': %s", asset_name, exc)
        raise NewsSearchError(f"Falha na pesquisa de notícias para '{asset_name}': {exc}") from exc
    except Exception as exc:
        logger.error("Erro inesperado na pesquisa de notícias para '%s': %s", asset_name, exc)
        raise NewsSearchError(f"Erro inesperado ao pesquisar notícias para '{asset_name}': {exc}") from exc

    if not raw_results:
        # Lista vazia não é necessariamente um erro (pode não haver notícias),
        # mas o Orquestrador precisa de saber que isto aconteceu para não
        # inventar um "sentimento" a partir de dados inexistentes.
        logger.warning("Nenhuma notícia encontrada para '%s'.", asset_name)

    noticias = [
        {
            "titulo": item.get("title"),
            "fonte": item.get("source"),
            "data_publicacao": item.get("date"),
            "url": item.get("url"),
            "excerto": item.get("body"),
        }
        for item in raw_results
    ]

    return {
        "ativo_pesquisado": asset_name,
        "query_utilizada": query,
        "num_noticias_encontradas": len(noticias),
        "noticias": noticias,
    }


if __name__ == "__main__":
    # Teste manual rápido do bloco 1 (sem dependência do CrewAI/LLM).
    import json

    for test_asset in ["Tesla", ""]:
        print(f"\n--- Teste: '{test_asset}' ---")
        try:
            result = get_recent_news(test_asset, max_results=5)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except NewsSearchError as e:
            print(f"[NewsSearchError] {e}")
