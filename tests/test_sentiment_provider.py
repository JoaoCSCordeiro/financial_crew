"""
Testes para o switch SENTIMENT_PROVIDER (llm | finetuned) -- ver
agents._news_analyst_tools, tasks._select_news_task_description e
tools/sentiment_tool.py.

Não constroem um `Agent`/`Task` real (isso exigiria um LLM configurado) --
testam diretamente as funções puras que decidem o comportamento consoante
a variável de ambiente, e a lógica de classificação em tools/sentiment_tool.py
com o HTTP mockado.
"""

from unittest.mock import patch

import pytest
import requests

from agents import _news_analyst_tools, _sentiment_provider
from tasks import (
    _NEWS_TASK_DESCRIPTION_FINETUNED,
    _NEWS_TASK_DESCRIPTION_LLM,
    _select_news_task_description,
)
from tools.exceptions import SentimentServiceError
from tools.sentiment_tool import classify_titles


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SENTIMENT_PROVIDER", raising=False)


class TestSentimentProviderDefault:
    def test_provider_default_e_llm(self):
        assert _sentiment_provider() == "llm"

    def test_tools_default_nao_inclui_classificador_finetuned(self):
        tools = _news_analyst_tools()
        assert [t.name for t in tools] == ["obter_noticias_recentes"]

    def test_task_default_usa_descricao_llm_original(self):
        assert _select_news_task_description() == _NEWS_TASK_DESCRIPTION_LLM
        assert "classificar_sentimento_titulos" not in _select_news_task_description()


class TestSentimentProviderFinetuned:
    def test_provider_finetuned_e_lido_do_ambiente(self, monkeypatch):
        monkeypatch.setenv("SENTIMENT_PROVIDER", "finetuned")
        assert _sentiment_provider() == "finetuned"

    def test_provider_e_case_insensitive_e_ignora_espacos(self, monkeypatch):
        monkeypatch.setenv("SENTIMENT_PROVIDER", "  FineTuned  ")
        assert _sentiment_provider() == "finetuned"

    def test_tools_finetuned_inclui_ambas_as_ferramentas(self, monkeypatch):
        monkeypatch.setenv("SENTIMENT_PROVIDER", "finetuned")
        tools = _news_analyst_tools()
        assert [t.name for t in tools] == [
            "obter_noticias_recentes",
            "classificar_sentimento_titulos",
        ]

    def test_task_finetuned_usa_descricao_que_delega_classificacao(self, monkeypatch):
        monkeypatch.setenv("SENTIMENT_PROVIDER", "finetuned")
        description = _select_news_task_description()
        assert description == _NEWS_TASK_DESCRIPTION_FINETUNED
        assert "classificar_sentimento_titulos" in description

    def test_valor_desconhecido_cai_no_comportamento_original(self, monkeypatch):
        monkeypatch.setenv("SENTIMENT_PROVIDER", "algo-invalido")
        assert _select_news_task_description() == _NEWS_TASK_DESCRIPTION_LLM
        assert [t.name for t in _news_analyst_tools()] == ["obter_noticias_recentes"]


class TestClassifyTitles:
    def test_lista_vazia_nao_chama_o_servico(self):
        result = classify_titles([])
        assert result == {
            "classificacoes": [],
            "positivos": 0,
            "negativos": 0,
            "neutros": 0,
            "sentimento_geral": None,
        }

    def _mock_response(self, labels):
        return [
            {"label": label, "confidence": 0.9, "probabilities": {}, "latency_ms": 1.0}
            for label in labels
        ]

    def test_regra_de_decisao_positivo(self):
        with patch("tools.sentiment_tool._call_classify_batch") as mock_call:
            mock_call.return_value = self._mock_response(
                ["positive", "positive", "negative"]
            )
            result = classify_titles(["A", "B", "C"])
        assert result["positivos"] == 2
        assert result["negativos"] == 1
        assert result["sentimento_geral"] == "positivo"

    def test_regra_de_decisao_misto_exige_empate_nao_nulo(self):
        with patch("tools.sentiment_tool._call_classify_batch") as mock_call:
            mock_call.return_value = self._mock_response(["positive", "negative"])
            result = classify_titles(["A", "B"])
        assert result["sentimento_geral"] == "misto"

    def test_regra_de_decisao_neutro_quando_tudo_neutro(self):
        with patch("tools.sentiment_tool._call_classify_batch") as mock_call:
            mock_call.return_value = self._mock_response(["neutral", "neutral"])
            result = classify_titles(["A", "B"])
        assert result["sentimento_geral"] == "neutro"

    def test_falha_de_conexao_apos_retries_levanta_sentiment_service_error(self):
        with patch("tools.sentiment_tool._call_classify_batch") as mock_call:
            mock_call.side_effect = requests.ConnectionError("recusado")
            with pytest.raises(SentimentServiceError):
                classify_titles(["A"])
