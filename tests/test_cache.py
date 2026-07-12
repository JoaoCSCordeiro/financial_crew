"""Testes para cache.py -- usa uma base de dados SQLite temporaria, nunca a real."""

import time

import cache
from schemas import AnaliseNoticias, DadosTecnicos, RelatorioFinanceiro


def _sample_report(ticker="TSLA", ativo="Tesla"):
    return RelatorioFinanceiro(
        ativo=ativo,
        ticker=ticker,
        dados_tecnicos=DadosTecnicos(
            preco_atual=100.0, moeda="USD", dados_indisponiveis=False
        ),
        analise_noticias=AnaliseNoticias(
            sentimento_geral="neutro", dados_indisponiveis=False
        ),
    )


def test_cache_miss_quando_vazia(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DB_PATH", tmp_path / "cache.db")
    assert cache.get_cached_report("TSLA") is None


def test_set_e_get_dentro_do_ttl(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DB_PATH", tmp_path / "cache.db")
    report = _sample_report()
    cache.set_cached_report("TSLA", "Tesla", report)

    cached = cache.get_cached_report("TSLA", ttl_seconds=60)
    assert cached is not None
    assert cached.ativo == "Tesla"
    assert cached.dados_tecnicos.preco_atual == 100.0


def test_get_expira_apos_ttl(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DB_PATH", tmp_path / "cache.db")
    report = _sample_report()
    cache.set_cached_report("TSLA", "Tesla", report)

    # TTL de 0 segundos -- qualquer entrada e imediatamente considerada expirada
    # (usamos um pequeno sleep para garantir que o relogio avancou).
    time.sleep(0.01)
    assert cache.get_cached_report("TSLA", ttl_seconds=0) is None


def test_set_substitui_entrada_existente(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DB_PATH", tmp_path / "cache.db")
    cache.set_cached_report(
        "TSLA", "Tesla", _sample_report(ticker="TSLA", ativo="Tesla")
    )

    report2 = _sample_report(ticker="TSLA", ativo="Tesla Motors")
    cache.set_cached_report("TSLA", "Tesla Motors", report2)

    cached = cache.get_cached_report("TSLA")
    assert cached.ativo == "Tesla Motors"


def test_tickers_diferentes_nao_se_confundem(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DB_PATH", tmp_path / "cache.db")
    cache.set_cached_report(
        "AAPL", "Apple", _sample_report(ticker="AAPL", ativo="Apple")
    )
    cache.set_cached_report(
        "TSLA", "Tesla", _sample_report(ticker="TSLA", ativo="Tesla")
    )

    assert cache.get_cached_report("AAPL").ativo == "Apple"
    assert cache.get_cached_report("TSLA").ativo == "Tesla"


def test_clear_cache_remove_tudo(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DB_PATH", tmp_path / "cache.db")
    cache.set_cached_report("AAPL", "Apple", _sample_report(ticker="AAPL"))
    cache.clear_cache()
    assert cache.get_cached_report("AAPL") is None


def test_get_com_json_corrompido_devolve_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DB_PATH", tmp_path / "cache.db")
    with cache._get_connection() as conn:
        conn.execute(
            "INSERT INTO analysis_cache (ticker, asset_name, report_json, cached_at) "
            "VALUES (?, ?, ?, ?)",
            ("AAPL", "Apple", "isto nao e json valido {{{", time.time()),
        )

    assert cache.get_cached_report("AAPL") is None
