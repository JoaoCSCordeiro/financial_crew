"""Testes para metrics.py -- usa um ficheiro temporario, nunca o metrics/metrics.jsonl real."""

import metrics


def test_record_and_read_single_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(metrics, "METRICS_FILE", tmp_path / "metrics.jsonl")

    metrics.record_analysis(
        "Apple",
        "AAPL",
        12.5,
        status="success",
        tokens={
            "total": 1000,
            "prompt": 700,
            "completion": 300,
            "successful_requests": 3,
        },
    )

    entries = metrics.read_metrics()
    assert len(entries) == 1
    assert entries[0]["asset_name"] == "Apple"
    assert entries[0]["ticker"] == "AAPL"
    assert entries[0]["status"] == "success"
    assert entries[0]["tokens"]["total"] == 1000


def test_read_metrics_devolve_lista_vazia_se_nao_existir_ficheiro(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(metrics, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(metrics, "METRICS_FILE", tmp_path / "nao_existe.jsonl")

    assert metrics.read_metrics() == []


def test_summarize_sem_execucoes(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(metrics, "METRICS_FILE", tmp_path / "vazio.jsonl")

    summary = metrics.summarize()
    assert summary["total_runs"] == 0
    assert summary["success_rate"] is None


def test_summarize_com_varias_execucoes(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(metrics, "METRICS_FILE", tmp_path / "metrics.jsonl")

    metrics.record_analysis("Apple", "AAPL", 10.0, "success", tokens={"total": 1000})
    metrics.record_analysis("Tesla", "TSLA", 30.0, "success", tokens={"total": 2000})
    metrics.record_analysis(
        "Bitcoin", "BTC-USD", 5.0, "error", error="RateLimitError: teste"
    )

    summary = metrics.summarize()
    assert summary["total_runs"] == 3
    assert summary["runs_by_status"] == {"success": 2, "error": 1}
    assert summary["success_rate"] == round(2 / 3, 3)
    assert summary["avg_duration_seconds"] == 15.0
    assert summary["total_tokens"] == 3000
    assert summary["slowest_run"]["ticker"] == "TSLA"
    assert summary["most_token_intensive_run"]["ticker"] == "TSLA"


def test_record_analysis_ignora_falha_de_escrita_sem_levantar(tmp_path, monkeypatch):
    """Uma falha ao escrever metricas nunca deve propagar (ver docstring de record_analysis)."""
    caminho_invalido = tmp_path / "sem_permissao" / "sub" / "metrics.jsonl"
    monkeypatch.setattr(metrics, "METRICS_DIR", tmp_path / "sem_permissao" / "sub")
    monkeypatch.setattr(metrics, "METRICS_FILE", caminho_invalido)

    # Cria um ficheiro no lugar onde o codigo esperava um diretorio, para
    # forcar mkdir(parents=True) a falhar com um OSError controlado.
    (tmp_path / "sem_permissao").write_text("isto devia ser uma pasta, nao um ficheiro")

    # Nao deve levantar excecao, apenas registar um aviso via logger.
    metrics.record_analysis("Apple", "AAPL", 1.0, "success")
