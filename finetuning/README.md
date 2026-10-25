# Fine-tuning — classificador de sentimento financeiro

Módulo opcional e independente do resto do Financial Crew: treina um
classificador de sentimento financeiro por LoRA sobre
`distilbert-base-uncased`, compara-o com a chamada LLM genérica que o
News Analyst usa hoje, e serve-o via FastAPI num container próprio. Ver
`../DECISOES-FINETUNING.md` (não versionado, preparação de entrevista)
para o raciocínio por trás de cada escolha, e `RESULTS.md` para os
números medidos.

Nada aqui corre automaticamente como parte do crew principal — é preciso
correr os passos abaixo explicitamente, e o crew só passa a usar o modelo
resultante se definires `SENTIMENT_PROVIDER=finetuned` (ver README da
raiz).

## Setup

```bash
pip install -r finetuning/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
# sem GPU: tira o --extra-index-url, o pip instala a build CPU-only
```

## 1. Treinar (3 experiências + registo + promoção)

```bash
python -m finetuning.run_experiments
```

Corre 3 configurações de LoRA (learning rate / rank / epochs diferentes),
regista cada uma no MLflow local (`finetuning/mlruns/`) e no Model
Registry (ver secção 1b), e exporta o adapter da versão `champion` para
`finetuning/models/best/`.

Ver o dashboard do MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///finetuning/mlruns/mlflow.db
```

## 1b. Model Registry (`registry.py`)

Cada run de treino passa a ser uma versão do modelo `financial-sentiment`,
empacotada como `pyfunc` (adapter LoRA + código de inferência, com
assinatura `text -> label, confidence`), com tags de proveniência
(run, modelo base, dataset, métricas).

Promoção por aliases (os stages Staging/Production foram descontinuados
no MLflow 2.9):

- `champion`: a versão que o serving usa. Uma versão nova só passa a
  `champion` se tiver `eval_f1_macro >= 0.90` (patamar do LLM genérico,
  ver RESULTS.md) e bater o campeão atual por pelo menos 0.005.
- `challenger`: passou o patamar, mas não bateu o campeão.
- `previous`: o campeão anterior, para rollback num passo.

As regras estão em `mlops_rules.py` (Python puro, testado na CI em
`tests/test_mlops_rules.py`).

```bash
python -m finetuning.registry register-runs   # regista runs antigos que ainda não têm versão
python -m finetuning.registry status          # versões, aliases e decisão do gate
python -m finetuning.registry export          # champion -> finetuning/models/best/ + MODEL_VERSION.json
python -m finetuning.registry rollback        # previous -> champion
```

Carregar qualquer versão sem saber o que está por baixo:

```python
mlflow.pyfunc.load_model("models:/financial-sentiment@champion").predict(df)
```

O serving não depende do MLflow: carrega o adapter exportado e o
`/health` devolve o `MODEL_VERSION.json` (versão, run, f1) para se saber
que versão está em produção.

## 1c. Monitorização de drift (`drift.py`)

```bash
python -m finetuning.drift --current phrasebank   # controlo: mesma distribuição
python -m finetuning.drift --current twitter      # shift real: tweets financeiros
python -m finetuning.drift --current log:finetuning/prediction_log.jsonl
```

Compara uma janela "current" com a referência (metade do test split do
Financial PhraseBank) usando o `DataDriftPreset` do Evidently sobre
features que não precisam de labels: forma do texto (caracteres,
palavras, tokens por palavra, caracteres não alfabéticos), confiança do
modelo e distribuição das previsões. Quando a janela tem labels
(`twitter`), mede também o f1 e a queda face ao campeão.

Cada execução fica na experiência `financial-sentiment-monitoring` do
MLflow (métricas por feature, `drift_share`, f1 da janela, relatório HTML
do Evidently como artefacto) e marca a versão `champion` com
`last_drift_check` e `retrain_recommended`. Retreino recomendado se
pelo menos 50% das features tiverem drift ou o f1 cair mais de 0.05.

Para monitorizar o tráfego real, arranca o serving com
`PREDICTION_LOG_PATH=finetuning/prediction_log.jsonl`: cada previsão
servida fica registada (texto, label, confiança, versão do modelo) e é
essa a janela do `--current log:...`. Resultados medidos: `RESULTS.md`,
secção 4.

## 2. Comparar com o LLM genérico (Groq)

```bash
python -m finetuning.compare --n-per-class 20
```

Precisa de `GROQ_API_KEY` real no `.env` da raiz (chama a API da Groq
diretamente via REST -- ver DECISOES-FINETUNING.md sobre porquê não usar
crewai/litellm aqui). Resultados medidos: `RESULTS.md`.

## 3. Servir localmente (sem Docker)

```bash
pip install -r finetuning/serving/requirements.txt
uvicorn finetuning.serving.app:app --host 0.0.0.0 --port 8001
```

```bash
curl -X POST http://localhost:8001/classify -H "Content-Type: application/json" \
  -d '{"text": "Company reports record quarterly profit"}'
```

## 4. Servir em Docker

Contexto de build tem de ser a raiz do repositório:

```bash
docker build -f finetuning/serving/Dockerfile -t financial-crew-sentiment .
docker run -p 8001:8001 financial-crew-sentiment
```

## 5. Ligar ao News Analyst

Na raiz do projeto, com o serviço de sentimento a correr (local ou
Docker) em `http://localhost:8001`:

```bash
SENTIMENT_PROVIDER=finetuned python main.py --ticker Apple:AAPL
```

`SENTIMENT_PROVIDER=llm` (ou a variável por omissão) mantém o
comportamento original — o LLM classifica os títulos ele próprio.
`SENTIMENT_SERVICE_URL` permite apontar para outro host/porta (default
`http://localhost:8001`).

## Estrutura

```
finetuning/
├── constants.py       # BASE_MODEL, labels -- sem imports pesados (usado pelo serving)
├── data.py             # carregamento/tokenização do Financial PhraseBank
├── train.py             # treino LoRA de uma configuração (TrainConfig -> RunResult)
├── tracking.py            # MLflow: configuração, seleção do melhor run
├── run_experiments.py      # corre as 3 configurações + regista + promove + exporta
├── registry.py              # Model Registry: pyfunc, gate de promoção, export, rollback
├── drift.py                  # monitorização de drift (Evidently) registada no MLflow
├── mlops_rules.py             # regras puras (promoção, features, retreino), testadas na CI
├── inference.py              # wrapper de inferência (usado por compare.py e serving/app.py)
├── compare.py                  # benchmark medido: fine-tuned vs. Groq LLM
├── RESULTS.md                    # números medidos (não versionar alterações sem remedir)
└── serving/
    ├── app.py                        # FastAPI
    ├── Dockerfile                     # imagem própria, contexto = raiz do repo
    └── requirements.txt                # torch CPU + transformers + peft + fastapi
```
