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

## 1. Treinar (3 experiências + seleção automática)

```bash
python -m finetuning.run_experiments
```

Corre 3 configurações de LoRA (learning rate / rank / epochs diferentes),
regista cada uma no MLflow local (`finetuning/mlruns/`), e copia o
adapter da melhor (por `eval_f1_macro`) para `finetuning/models/best/`.

Ver o dashboard do MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///finetuning/mlruns/mlflow.db
```

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
├── run_experiments.py      # corre as 3 configurações + seleciona a melhor
├── inference.py              # wrapper de inferência (usado por compare.py e serving/app.py)
├── compare.py                  # benchmark medido: fine-tuned vs. Groq LLM
├── RESULTS.md                    # números medidos (não versionar alterações sem remedir)
└── serving/
    ├── app.py                        # FastAPI
    ├── Dockerfile                     # imagem própria, contexto = raiz do repo
    └── requirements.txt                # torch CPU + transformers + peft + fastapi
```
