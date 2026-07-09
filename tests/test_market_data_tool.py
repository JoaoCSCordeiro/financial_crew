"""Testes para tools/market_data_tool.py -- yfinance e mockado, nunca chamado a serio."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tools.exceptions import InvalidTickerError, MarketDataUnavailableError
from tools.market_data_tool import get_market_data


def _fake_history_df():
    """Constroi um DataFrame minimo parecido com o devolvido por yf.Ticker.history()."""
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 103.0],
            "High": [105.0, 106.0, 108.0],
            "Low": [98.0, 99.0, 101.0],
            "Close": [102.0, 104.0, 107.0],
            "Volume": [1_000_000, 1_100_000, 1_200_000],
        }
    )


class TestGetMarketDataSuccess:
    @patch("tools.market_data_tool._fetch_ticker_object")
    def test_devolve_dados_estruturados_corretos(self, mock_fetch):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _fake_history_df()
        mock_ticker.info = {
            "currentPrice": 107.5,
            "currency": "USD",
            "shortName": "Fake Corp",
        }
        mock_fetch.return_value = mock_ticker

        result = get_market_data("FAKE")

        assert result["ticker"] == "FAKE"
        assert result["preco_atual"] == 107.5
        assert result["moeda"] == "USD"
        assert result["nome_empresa"] == "Fake Corp"
        assert result["historico_1_mes"]["primeiro_fecho"] == 102.0
        assert result["historico_1_mes"]["ultimo_fecho"] == 107.0
        assert result["historico_1_mes"]["maximo"] == 108.0
        assert result["historico_1_mes"]["minimo"] == 98.0
        assert result["historico_1_mes"]["num_sessoes"] == 3

    @patch("tools.market_data_tool._fetch_ticker_object")
    def test_usa_ultimo_fecho_quando_info_falha(self, mock_fetch):
        """Se yf_ticker.info lançar exceção, deve usar o último fecho como preço."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _fake_history_df()
        type(mock_ticker).info = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        mock_fetch.return_value = mock_ticker

        result = get_market_data("FAKE")

        assert result["preco_atual"] == 107.0  # ultimo fecho, fallback
        assert result["moeda"] == "USD"  # default

    @patch("tools.market_data_tool._fetch_ticker_object")
    def test_ticker_e_normalizado_para_maiusculas(self, mock_fetch):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _fake_history_df()
        mock_ticker.info = {"currentPrice": 100.0}
        mock_fetch.return_value = mock_ticker

        result = get_market_data("  fake  ")

        assert result["ticker"] == "FAKE"


class TestGetMarketDataErrors:
    def test_ticker_vazio_levanta_invalid_ticker_error(self):
        with pytest.raises(InvalidTickerError):
            get_market_data("")

    def test_ticker_so_espacos_levanta_invalid_ticker_error(self):
        with pytest.raises(InvalidTickerError):
            get_market_data("   ")

    @patch("tools.market_data_tool._fetch_ticker_object")
    def test_historico_vazio_levanta_invalid_ticker_error(self, mock_fetch):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()  # sem dados
        mock_fetch.return_value = mock_ticker

        with pytest.raises(InvalidTickerError):
            get_market_data("TICKER_INEXISTENTE")

    @patch("tools.market_data_tool._fetch_ticker_object")
    def test_erro_de_rede_levanta_market_data_unavailable_error(self, mock_fetch):
        mock_fetch.side_effect = ConnectionError("falha de rede simulada")

        with pytest.raises(MarketDataUnavailableError):
            get_market_data("AAPL")

    @patch("tools.market_data_tool._fetch_ticker_object")
    def test_erro_inesperado_do_yfinance_e_convertido(self, mock_fetch):
        """Exceções não tipadas do yfinance devem ser convertidas, nunca propagar em bruto."""
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = ValueError(
            "erro interno estranho do yfinance"
        )
        mock_fetch.return_value = mock_ticker

        with pytest.raises(MarketDataUnavailableError):
            get_market_data("AAPL")
