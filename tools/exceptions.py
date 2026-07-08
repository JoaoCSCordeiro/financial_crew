"""
Exceções customizadas para o sistema multi-agente.

Ter exceções específicas (em vez de `Exception` genérica) permite que o
Agente Orquestrador (Bloco 3) distinga entre falhas recuperáveis (ex: rede)
e falhas de dados (ex: ticker inválido), tomando decisões diferentes em
cada caso, e evita que erros sejam mascarados como "sucesso silencioso".
"""


class FinancialToolError(Exception):
    """Classe base para todos os erros lançados pelas ferramentas do sistema."""


class InvalidTickerError(FinancialToolError):
    """Lançado quando o ticker não existe ou a Yahoo Finance não devolve dados."""


class MarketDataUnavailableError(FinancialToolError):
    """Lançado quando a rede falha ou a API do Yahoo Finance está indisponível."""


class NewsSearchError(FinancialToolError):
    """Lançado quando a pesquisa de notícias (DuckDuckGo) falha ou é limitada (rate limit)."""
