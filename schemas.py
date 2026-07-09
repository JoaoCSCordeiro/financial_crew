"""
Schema de Output - Bloco 5.

Define a estrutura estrita que o relatorio final do Agente 3 (Orquestrador)
tem de respeitar. Passar isto como `output_pydantic` na Task (ver tasks.py)
faz o CrewAI validar e forcar o output do LLM contra este schema -- em vez
de confiarmos que o LLM "escreveu JSON valido" (o que falha ocasionalmente),
o CrewAI tenta converter/corrigir o output para respeitar este contrato.

Isto e o que torna o resultado seguro para um frontend ou base de dados
consumir diretamente, sem parsing fragil de string.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DadosTecnicos(BaseModel):
    """Dados de mercado consolidados, rastreaveis ate a ferramenta yfinance."""

    preco_atual: Optional[float] = Field(
        default=None,
        description="Preco atual do ativo, exatamente como devolvido pela ferramenta.",
    )
    moeda: Optional[str] = Field(
        default=None, description="Moeda do preco (ex: 'USD')."
    )
    variacao_percentual_1_mes: Optional[float] = Field(
        default=None, description="Variacao percentual no ultimo mes."
    )
    tendencia: Optional[Literal["alta", "baixa", "lateral"]] = Field(
        default=None,
        description="Tendencia identificada com base no historico de 1 mes.",
    )
    dados_indisponiveis: bool = Field(
        default=False,
        description="True se a ferramenta de dados de mercado falhou ou nao devolveu dados utilizaveis.",
    )


class AnaliseNoticias(BaseModel):
    """Analise de sentimento de noticias, rastreavel ate a ferramenta ddgs."""

    sentimento_geral: Optional[Literal["positivo", "negativo", "neutro", "misto"]] = (
        Field(
            default=None,
            description="Sentimento geral identificado a partir das noticias recolhidas.",
        )
    )
    titulos_usados: list[str] = Field(
        default_factory=list,
        description="Titulos de noticias usados como evidencia, exatamente como devolvidos pela ferramenta.",
    )
    dados_indisponiveis: bool = Field(
        default=False,
        description="True se a pesquisa de noticias falhou ou nao devolveu resultados.",
    )


class RelatorioFinanceiro(BaseModel):
    """
    Relatorio final consolidado pelo Agente 3 (Orquestrador/Critico).

    Esta e a unica estrutura que sai do sistema -- qualquer consumidor
    (frontend, base de dados, outro servico) pode depender destes campos
    e tipos sem se preocupar com variacao de formato entre execucoes.
    """

    ativo: str = Field(description="Nome do ativo em linguagem natural (ex: 'Tesla').")
    ticker: str = Field(description="Simbolo do ativo (ex: 'TSLA').")
    dados_tecnicos: DadosTecnicos
    analise_noticias: AnaliseNoticias
    notas_de_validacao: str = Field(
        default="",
        description=(
            "Contradicoes, dados em falta, ou problemas encontrados pelo "
            "Orquestrador durante a validacao. String vazia se nao houver problemas."
        ),
    )
