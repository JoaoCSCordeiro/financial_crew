"""
Testes para crew._extract_json_report -- a funcao que substitui output_pydantic
(ver README, seccao "Known Issues" sobre o bug de tool-calling da Groq).

Estes testes nao tocam na rede nem no LLM: alimentam texto em bruto tal
como um LLM o poderia devolver, e confirmam que o parse+validacao funciona
nos casos realistas (incluindo os "maus" que motivaram esta abordagem).
"""

from crew import _extract_json_report, describe_rate_limit_error

VALID_JSON = """{
  "ativo": "Apple",
  "ticker": "AAPL",
  "dados_tecnicos": {
    "preco_atual": 313.39,
    "moeda": "USD",
    "variacao_percentual_1_mes": 7.86,
    "tendencia": "alta",
    "dados_indisponiveis": false
  },
  "analise_noticias": {
    "sentimento_geral": "positivo",
    "titulos_usados": ["Título A", "Título B"],
    "dados_indisponiveis": false
  },
  "notas_de_validacao": ""
}"""


class TestExtractJsonReportSucesso:
    def test_extrai_json_puro(self):
        report = _extract_json_report(VALID_JSON)
        assert report is not None
        assert report.ativo == "Apple"
        assert report.ticker == "AAPL"

    def test_extrai_json_com_texto_a_volta(self):
        """Caso real observado: o LLM por vezes escreve uma frase antes/depois do JSON."""
        raw = f"Aqui está o relatório final consolidado:\n\n{VALID_JSON}\n\nEspero que ajude!"
        report = _extract_json_report(raw)
        assert report is not None
        assert report.ativo == "Apple"

    def test_extrai_json_com_markdown_fences(self):
        raw = f"```json\n{VALID_JSON}\n```"
        report = _extract_json_report(raw)
        assert report is not None
        assert report.ticker == "AAPL"


class TestExtractJsonReportFalha:
    def test_texto_sem_json_devolve_none(self):
        assert _extract_json_report("Isto não contém nenhum JSON.") is None

    def test_json_invalido_sintaticamente_devolve_none(self):
        raw = '{"ativo": "Apple", "ticker": }'  # JSON mal formado
        assert _extract_json_report(raw) is None

    def test_json_valido_mas_fora_do_schema_devolve_none(self):
        """JSON sintaticamente válido, mas sem os campos obrigatórios do schema."""
        raw = '{"apenas_um_campo": "qualquer coisa"}'
        assert _extract_json_report(raw) is None

    def test_json_com_tendencia_invalida_devolve_none(self):
        raw = VALID_JSON.replace('"tendencia": "alta"', '"tendencia": "explosiva"')
        assert _extract_json_report(raw) is None

    def test_string_vazia_devolve_none(self):
        assert _extract_json_report("") is None


class TestDescribeRateLimitError:
    def test_distingue_limite_diario(self):
        exc = Exception(
            'RateLimitError: GroqException - {"error":{"message":"Rate limit reached '
            'for model on tokens per day (TPD): Limit 200000"}}'
        )
        description = describe_rate_limit_error(exc)
        assert "DIÁRIO" in description
        assert "200 000" in description

    def test_distingue_limite_por_minuto(self):
        exc = Exception(
            'RateLimitError: GroqException - {"error":{"message":"Rate limit reached '
            'for model on tokens per minute (TPM): Limit 8000"}}'
        )
        description = describe_rate_limit_error(exc)
        assert "POR MINUTO" in description
        assert "8000" in description

    def test_fallback_generico_quando_tipo_nao_identificado(self):
        exc = Exception("rate limit exceeded, no further details")
        description = describe_rate_limit_error(exc)
        assert "Rate limit da Groq excedido" in description

    def test_segue_a_cadeia_de_causas(self):
        cause = Exception("rate limit exceeded: tokens per day (TPD) limit hit")
        wrapper = Exception("InstructorRetryException: falhou")
        wrapper.__cause__ = cause

        description = describe_rate_limit_error(wrapper)
        assert "DIÁRIO" in description
