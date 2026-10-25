"""
Regras de decisão do ciclo de vida do modelo (promoção no Model Registry,
features de monitorização e recomendação de retreino).

Deliberadamente em Python puro (sem `mlflow`, `torch`, `evidently` nem
`pandas`): é a parte que mais interessa testar e é a única que a CI
consegue testar, porque a CI só instala o requirements.txt da raiz (ver
.github/workflows/ci.yml). `registry.py` e `drift.py` tratam do I/O
(MLflow, Evidently, modelo) e delegam aqui todas as decisões.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Promoção no Model Registry ---------------------------------------------

# Abaixo disto o candidato é rejeitado mesmo que não haja campeão: é o
# patamar do LLM genérico medido em RESULTS.md (90% de accuracy), e um
# modelo fine-tuned que não o bate não tem razão para ir para produção.
MIN_F1_MACRO = 0.90
# Margem mínima para substituir o campeão. Sem ela, duas corridas com
# diferenças de ruído (décimas de ponto no mesmo test set) estariam
# sempre a trocar de campeão.
MIN_PROMOTION_DELTA = 0.005

PROMOTE = "promote"
KEEP_CHAMPION = "keep_champion"
REJECT = "reject"


@dataclass(frozen=True)
class PromotionDecision:
    action: str
    reason: str


def decide_promotion(
    candidate_f1: float,
    champion_f1: float | None,
    min_f1: float = MIN_F1_MACRO,
    min_delta: float = MIN_PROMOTION_DELTA,
) -> PromotionDecision:
    """
    Decide o destino de uma nova versão registada.

    - REJECT: não passa o patamar absoluto (`min_f1`).
    - PROMOTE: não há campeão, ou bate o campeão por pelo menos `min_delta`.
    - KEEP_CHAMPION: passa o patamar mas não bate o campeão por margem
      suficiente; fica como "challenger".
    """
    if candidate_f1 < min_f1:
        return PromotionDecision(
            REJECT, f"f1_macro {candidate_f1:.4f} abaixo do mínimo {min_f1:.2f}"
        )
    if champion_f1 is None:
        return PromotionDecision(PROMOTE, "não existe campeão")
    if candidate_f1 >= champion_f1 + min_delta:
        return PromotionDecision(
            PROMOTE,
            f"f1_macro {candidate_f1:.4f} >= campeão {champion_f1:.4f} + {min_delta}",
        )
    return PromotionDecision(
        KEEP_CHAMPION,
        f"f1_macro {candidate_f1:.4f} não bate o campeão {champion_f1:.4f} "
        f"por {min_delta}",
    )


# --- Features de monitorização ----------------------------------------------

# Features numéricas calculadas por texto. Não há labels em produção, por
# isso a monitorização olha para (1) a forma do input e (2) o
# comportamento do modelo (confiança e distribuição das previsões).
NUMERICAL_FEATURES = [
    "char_length",
    "word_count",
    "tokens_per_word",
    "non_alpha_ratio",
    "confidence",
]
CATEGORICAL_FEATURES = ["predicted_label"]


def text_features(text: str, n_tokens: int) -> dict:
    """
    Features de forma do texto.

    `n_tokens` vem do tokenizer do modelo (sem tokens especiais): um
    `tokens_per_word` alto quer dizer que o vocabulário do input está longe
    do que o modelo viu (tickers, URLs, calão), o que costuma anteceder a
    queda de qualidade antes de haver labels para a medir.
    """
    words = text.split()
    n_chars = len(text)
    non_alpha = sum(1 for c in text if not (c.isalpha() or c.isspace()))
    return {
        "char_length": n_chars,
        "word_count": len(words),
        "tokens_per_word": n_tokens / len(words) if words else 0.0,
        "non_alpha_ratio": non_alpha / n_chars if n_chars else 0.0,
    }


# --- Dataset de shift: Twitter Financial News --------------------------------

# zeroshot/twitter-financial-news-sentiment usa 0=Bearish, 1=Bullish,
# 2=Neutral; o modelo usa 0=negative, 1=neutral, 2=positive
# (constants.LABEL_NAMES).
TWITTER_TO_MODEL_LABEL = {0: 0, 1: 2, 2: 1}


# --- Leitura do relatório do Evidently ----------------------------------------


@dataclass(frozen=True)
class ColumnDrift:
    column: str
    method: str
    score: float
    threshold: float
    drifted: bool


def parse_value_drift(metric: dict) -> ColumnDrift | None:
    """
    Converte uma entrada `ValueDrift` do `Snapshot.dict()` do Evidently 0.7
    num `ColumnDrift`; devolve None para as outras métricas.

    Os testes estatísticos devolvem p-values (drift se `score < threshold`),
    as distâncias (Wasserstein, Jensen-Shannon, PSI) devolvem uma distância
    (drift se `score >= threshold`). O Evidently escolhe o método pelo tipo
    e tamanho da coluna, por isso a direção vem do nome do método.
    """
    config = metric.get("config", {})
    if not config.get("type", "").endswith("ValueDrift"):
        return None
    method = config.get("method", "")
    threshold = float(config["threshold"])
    score = float(metric["value"])
    drifted = score < threshold if "p_value" in method else score >= threshold
    return ColumnDrift(config["column"], method, score, threshold, drifted)


# --- Recomendação de retreino --------------------------------------------------

# Metade das features com drift é o mesmo critério do `DataDriftPreset`
# (drift_share=0.5) para declarar drift no dataset inteiro.
MAX_DRIFT_SHARE = 0.5
# Queda de f1_macro (em pontos absolutos) face à avaliação do campeão a
# partir da qual o modelo deixa de servir, quando há labels.
MAX_F1_DROP = 0.05


def recommend_retrain(
    drift_share: float,
    f1_drop: float | None,
    max_drift_share: float = MAX_DRIFT_SHARE,
    max_f1_drop: float = MAX_F1_DROP,
) -> tuple[bool, list[str]]:
    """
    Junta o sinal sem labels (drift) e o sinal com labels (queda de f1,
    quando existe) numa recomendação de retreino com as razões.
    """
    reasons = []
    if drift_share >= max_drift_share:
        reasons.append(
            f"{drift_share:.0%} das features com drift (limite {max_drift_share:.0%})"
        )
    if f1_drop is not None and f1_drop > max_f1_drop:
        reasons.append(
            f"f1_macro caiu {f1_drop:.3f} face ao campeão (limite {max_f1_drop})"
        )
    return bool(reasons), reasons
