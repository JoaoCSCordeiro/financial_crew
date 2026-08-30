"""
Constantes partilhadas entre treino, avaliação e serving.

Deliberadamente sem imports pesados (sem `torch`/`transformers`/`datasets`/
`sklearn`): o serving (`finetuning/serving/app.py`, via `inference.py`)
importa este módulo e só este módulo do lado dos metadados do dataset --
não pode arrastar consigo dependências só-de-treino (`datasets`, `mlflow`,
`scikit-learn`) para dentro do container de serving, que tem um
requirements.txt propositadamente mais leve (ver
finetuning/serving/requirements.txt).
"""

BASE_MODEL = "distilbert-base-uncased"

LABEL_NAMES = ["negative", "neutral", "positive"]
ID2LABEL = dict(enumerate(LABEL_NAMES))
LABEL2ID = {name: idx for idx, name in ID2LABEL.items()}
MAX_LENGTH = 128
