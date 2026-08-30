"""
Adaptadores CrewAI para as ferramentas de negócio.

Camada fina de integração: traduz `FinancialToolError` em texto legível
para o LLM (os agentes não sabem lidar com exceções Python — só com
strings), sem nunca esconder que algo falhou. Isto é crítico: se
devolvermos um erro disfarçado de sucesso, o Orquestrador (Bloco 3) pode
alucinar em cima de dados inexistentes.
"""

import json
from typing import List

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from tools.exceptions import FinancialToolError
from tools.market_data_tool import MarketDataSchema, get_market_data
from tools.news_tool import NewsSearchSchema, get_recent_news
from tools.sentiment_tool import classify_titles


class MarketDataTool(BaseTool):
    name: str = "obter_dados_mercado"
    description: str = (
        "Obtém o preço atual e o histórico de 1 mês de um ativo financeiro via Yahoo Finance. "
        "Input: o ticker do ativo (ex: 'AAPL', 'TSLA', 'BTC-USD', 'ETH-USD', 'MSFT'). "
        "Devolve dados JSON reais e verificáveis — nunca inventes números que não venham desta ferramenta."
    )
    args_schema: type[MarketDataSchema] = MarketDataSchema

    def _run(self, ticker: str) -> str:
        try:
            data = get_market_data(ticker)
            return json.dumps(data, ensure_ascii=False)
        except FinancialToolError as exc:
            # Devolvemos o erro como string estruturada e explícita para o agente,
            # em vez de deixar a exceção propagar e crashar o Crew inteiro.
            return json.dumps(
                {"erro": True, "tipo": type(exc).__name__, "mensagem": str(exc)},
                ensure_ascii=False,
            )


class NewsSearchTool(BaseTool):
    name: str = "obter_noticias_recentes"
    description: str = (
        "Obtém os títulos de notícias mais recentes sobre um ativo financeiro via DuckDuckGo. "
        "Input: o nome do ativo em linguagem natural (ex: 'Tesla', 'Bitcoin', 'Apple'). "
        "Devolve títulos e fontes reais — nunca inventes ou parafraseies notícias que não estejam aqui."
    )
    args_schema: type[NewsSearchSchema] = NewsSearchSchema

    def _run(self, asset_name: str, max_results: int = 5) -> str:
        try:
            data = get_recent_news(asset_name, max_results=max_results)
            return json.dumps(data, ensure_ascii=False)
        except FinancialToolError as exc:
            return json.dumps(
                {"erro": True, "tipo": type(exc).__name__, "mensagem": str(exc)},
                ensure_ascii=False,
            )


class SentimentClassifierSchema(BaseModel):
    """Schema de input validado para a ferramenta de classificação fine-tuned."""

    titulos: List[str] = Field(
        ...,
        description="Lista de títulos de notícias a classificar (exatamente como devolvidos por 'obter_noticias_recentes').",
    )


class FineTunedSentimentTool(BaseTool):
    name: str = "classificar_sentimento_titulos"
    description: str = (
        "Classifica o sentimento (positivo/negativo/neutro) de uma lista de títulos de "
        "notícias usando o modelo fine-tuned local (LoRA sobre distilbert-base-uncased), "
        "e devolve também a contagem agregada e o sentimento geral já calculado pela regra "
        "de decisão P vs N. Usa esta ferramenta em vez de classificares os títulos por ti "
        "próprio -- é mais rápida e mais consistente."
    )
    args_schema: type[SentimentClassifierSchema] = SentimentClassifierSchema

    def _run(self, titulos: List[str]) -> str:
        try:
            data = classify_titles(titulos)
            return json.dumps(data, ensure_ascii=False)
        except FinancialToolError as exc:
            return json.dumps(
                {"erro": True, "tipo": type(exc).__name__, "mensagem": str(exc)},
                ensure_ascii=False,
            )
