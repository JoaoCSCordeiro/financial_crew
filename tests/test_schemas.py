"""Testes para schemas.py -- validacao Pydantic do relatorio final."""

import pytest
from pydantic import ValidationError

from schemas import AnaliseNoticias, DadosTecnicos, RelatorioFinanceiro


class TestRelatorioFinanceiroValido:
    def test_aceita_relatorio_completo_valido(self):
        relatorio = RelatorioFinanceiro(
            ativo="Tesla",
            ticker="TSLA",
            dados_tecnicos=DadosTecnicos(
                preco_atual=392.88,
                moeda="USD",
                variacao_percentual_1_mes=-3.93,
                tendencia="baixa",
                dados_indisponiveis=False,
            ),
            analise_noticias=AnaliseNoticias(
                sentimento_geral="misto",
                titulos_usados=["Título A", "Título B"],
                dados_indisponiveis=False,
            ),
            notas_de_validacao="",
        )
        assert relatorio.ativo == "Tesla"
        assert relatorio.dados_tecnicos.tendencia == "baixa"

    def test_aceita_dados_indisponiveis_com_campos_nulos(self):
        """Quando uma ferramenta falha, os campos numéricos devem poder ser None."""
        relatorio = RelatorioFinanceiro(
            ativo="Apple",
            ticker="AAPL",
            dados_tecnicos=DadosTecnicos(dados_indisponiveis=True),
            analise_noticias=AnaliseNoticias(dados_indisponiveis=True),
        )
        assert relatorio.dados_tecnicos.preco_atual is None
        assert relatorio.dados_tecnicos.tendencia is None
        assert relatorio.analise_noticias.titulos_usados == []
        assert relatorio.notas_de_validacao == ""  # default

    def test_model_validate_a_partir_de_dict_como_no_crew_py(self):
        """Reproduz o caminho real: JSON parseado de texto e validado com model_validate."""
        data = {
            "ativo": "Bitcoin",
            "ticker": "BTC-USD",
            "dados_tecnicos": {
                "preco_atual": 62000.5,
                "moeda": "USD",
                "variacao_percentual_1_mes": -1.5,
                "tendencia": "baixa",
                "dados_indisponiveis": False,
            },
            "analise_noticias": {
                "sentimento_geral": "negativo",
                "titulos_usados": ["Bitcoin cai"],
                "dados_indisponiveis": False,
            },
            "notas_de_validacao": "",
        }
        relatorio = RelatorioFinanceiro.model_validate(data)
        assert relatorio.ticker == "BTC-USD"
        assert relatorio.analise_noticias.sentimento_geral == "negativo"


class TestRelatorioFinanceiroInvalido:
    def test_rejeita_tendencia_fora_do_enum(self):
        with pytest.raises(ValidationError):
            DadosTecnicos(tendencia="a_subir_muito")  # não é um valor válido

    def test_rejeita_sentimento_fora_do_enum(self):
        with pytest.raises(ValidationError):
            AnaliseNoticias(sentimento_geral="super_positivo")

    def test_rejeita_relatorio_sem_campos_obrigatorios(self):
        with pytest.raises(ValidationError):
            RelatorioFinanceiro(
                ativo="Tesla"
            )  # falta ticker, dados_tecnicos, analise_noticias

    def test_rejeita_ativo_nao_string(self):
        with pytest.raises(ValidationError):
            RelatorioFinanceiro.model_validate(
                {
                    "ativo": 123,  # deveria ser string
                    "ticker": "TSLA",
                    "dados_tecnicos": {},
                    "analise_noticias": {},
                }
            )
