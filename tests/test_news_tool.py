"""Testes para tools/news_tool.py -- ddgs (DuckDuckGo) e mockado, nunca chamado a serio."""

from unittest.mock import patch

import pytest
from ddgs.exceptions import RatelimitException

from tools.exceptions import NewsSearchError
from tools.news_tool import get_recent_news


def _fake_ddg_results():
    return [
        {
            "title": "Título de teste 1",
            "source": "Fonte Teste",
            "date": "2026-07-01T10:00:00+00:00",
            "url": "https://example.com/1",
            "body": "Excerto de teste 1",
        },
        {
            "title": "Título de teste 2",
            "source": "Outra Fonte",
            "date": "2026-07-02T10:00:00+00:00",
            "url": "https://example.com/2",
            "body": "Excerto de teste 2",
        },
    ]


class TestGetRecentNewsSuccess:
    @patch("tools.news_tool._run_ddg_news_search")
    def test_devolve_noticias_estruturadas_corretamente(self, mock_search):
        mock_search.return_value = _fake_ddg_results()

        result = get_recent_news("Tesla", max_results=5)

        assert result["ativo_pesquisado"] == "Tesla"
        assert result["num_noticias_encontradas"] == 2
        assert result["noticias"][0]["titulo"] == "Título de teste 1"
        assert result["noticias"][0]["fonte"] == "Fonte Teste"
        assert result["noticias"][1]["titulo"] == "Título de teste 2"

    @patch("tools.news_tool._run_ddg_news_search")
    def test_zero_noticias_nao_e_erro(self, mock_search):
        """Lista vazia é um resultado válido (não há notícias), não uma exceção."""
        mock_search.return_value = []

        result = get_recent_news("AtivoObscuro")

        assert result["num_noticias_encontradas"] == 0
        assert result["noticias"] == []

    @patch("tools.news_tool._run_ddg_news_search")
    def test_query_construida_inclui_nome_do_ativo(self, mock_search):
        mock_search.return_value = []
        get_recent_news("Bitcoin")

        called_query = mock_search.call_args[0][0]
        assert "Bitcoin" in called_query


class TestGetRecentNewsErrors:
    def test_nome_vazio_levanta_news_search_error(self):
        with pytest.raises(NewsSearchError):
            get_recent_news("")

    def test_nome_so_espacos_levanta_news_search_error(self):
        with pytest.raises(NewsSearchError):
            get_recent_news("   ")

    @patch("tools.news_tool._run_ddg_news_search")
    def test_rate_limit_e_convertido_em_news_search_error(self, mock_search):
        mock_search.side_effect = RatelimitException("rate limit simulado")

        with pytest.raises(NewsSearchError):
            get_recent_news("Tesla")

    @patch("tools.news_tool._run_ddg_news_search")
    def test_erro_generico_e_convertido_em_news_search_error(self, mock_search):
        mock_search.side_effect = RuntimeError("erro inesperado qualquer")

        with pytest.raises(NewsSearchError):
            get_recent_news("Tesla")
