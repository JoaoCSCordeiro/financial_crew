"""Testes para finetuning/mlops_rules.py -- promoção no registry, features e drift."""

from finetuning.mlops_rules import (
    KEEP_CHAMPION,
    PROMOTE,
    REJECT,
    TWITTER_TO_MODEL_LABEL,
    decide_promotion,
    parse_value_drift,
    recommend_retrain,
    text_features,
)


def test_promove_quando_nao_ha_campeao():
    assert decide_promotion(0.93, None).action == PROMOTE


def test_rejeita_abaixo_do_minimo_mesmo_sem_campeao():
    assert decide_promotion(0.85, None).action == REJECT


def test_promove_quando_bate_campeao_pela_margem():
    assert decide_promotion(0.9564, 0.9349).action == PROMOTE


def test_mantem_campeao_quando_diferenca_e_ruido():
    # run-c (0.9535) contra run-b (0.9564): pior, fica como challenger
    assert decide_promotion(0.9535, 0.9564).action == KEEP_CHAMPION
    # ligeiramente melhor, mas abaixo da margem de 0.005
    assert decide_promotion(0.9580, 0.9564).action == KEEP_CHAMPION


def test_text_features_conta_forma_do_texto():
    f = text_features("$AAPL up 5% https://t.co/x", n_tokens=12)
    assert f["word_count"] == 4
    assert f["char_length"] == 26
    assert f["tokens_per_word"] == 3.0
    assert 0 < f["non_alpha_ratio"] < 1


def test_text_features_texto_vazio_nao_divide_por_zero():
    f = text_features("", n_tokens=0)
    assert f["tokens_per_word"] == 0.0
    assert f["non_alpha_ratio"] == 0.0


def test_mapeamento_labels_twitter():
    # Twitter: 0=Bearish, 1=Bullish, 2=Neutral -> modelo: 0=negative, 1=neutral, 2=positive
    assert TWITTER_TO_MODEL_LABEL == {0: 0, 1: 2, 2: 1}


def _value_drift(method: str, value: float, threshold: float) -> dict:
    return {
        "config": {
            "type": "evidently:metric_v2:ValueDrift",
            "column": "confidence",
            "method": method,
            "threshold": threshold,
        },
        "value": value,
    }


def test_parse_value_drift_p_value_baixo_e_drift():
    assert parse_value_drift(_value_drift("K-S p_value", 0.001, 0.05)).drifted
    assert not parse_value_drift(_value_drift("K-S p_value", 0.4, 0.05)).drifted


def test_parse_value_drift_distancia_alta_e_drift():
    assert parse_value_drift(
        _value_drift("Wasserstein distance (normed)", 0.3, 0.1)
    ).drifted
    assert not parse_value_drift(
        _value_drift("Wasserstein distance (normed)", 0.02, 0.1)
    ).drifted


def test_parse_value_drift_ignora_outras_metricas():
    metric = {
        "config": {"type": "evidently:metric_v2:DriftedColumnsCount"},
        "value": {"count": 2.0, "share": 1.0},
    }
    assert parse_value_drift(metric) is None


def test_recomenda_retreino_por_drift_ou_queda_de_f1():
    assert recommend_retrain(1.0, 0.31)[0]
    assert recommend_retrain(0.6, None)[0]
    assert recommend_retrain(0.0, 0.08)[0]


def test_nao_recomenda_retreino_sem_sinais():
    retrain, reasons = recommend_retrain(0.0, -0.01)
    assert not retrain
    assert reasons == []
