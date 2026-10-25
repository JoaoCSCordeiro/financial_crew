# Resultados medidos — fine-tuning de sentimento financeiro

Todos os números abaixo foram medidos nesta máquina (CPU: local, GPU: NVIDIA
GeForce GTX 1650 Ti, 4GB VRAM) em 2026-08-30, correndo os scripts deste
módulo tal como estão no repositório. Nenhum número é estimado — onde um
cálculo depende de um preço público (tarifa Groq), isso está identificado
explicitamente. Para reproduzir:

```bash
pip install -r finetuning/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
python -m finetuning.run_experiments   # ~7 min em GPU, gera a tabela abaixo
python -m finetuning.compare --n-per-class 20   # ~2-3 min, precisa de GROQ_API_KEY real
```

## 1. Experiências de fine-tuning (MLflow)

Dataset: Financial PhraseBank (`FinanceMTEB/financial_phrasebank`, subset
`sentences_allagree`), split de teste fixo com 1000 frases (612 neutras,
250 positivas, 138 negativas — ver DECISOES-FINETUNING.md sobre o desequilíbrio
de classes e a escolha de `f1_macro` como métrica de seleção).

| Run | learning_rate | LoRA r | LoRA alpha | epochs | eval_accuracy | eval_f1_macro | eval_precision_macro | eval_recall_macro | eval_loss | tempo de treino |
|---|---|---|---|---|---|---|---|---|---|---|
| run-a_lr1e-3_r4_ep3 | 1e-3 | 4 | 8 | 3 | 0.9540 | 0.9349 | 0.9339 | 0.9362 | 0.1416 | 88.0s |
| **run-b_lr5e-4_r8_ep5 (melhor)** | 5e-4 | 8 | 16 | 5 | **0.9680** | **0.9564** | 0.9519 | 0.9611 | 0.1381 | 144.8s |
| run-c_lr2e-4_r16_ep5 | 2e-4 | 16 | 32 | 5 | 0.9650 | 0.9535 | 0.9496 | 0.9576 | 0.1071 | 146.2s |

Seleção automática (`finetuning/tracking.select_best_run`, por
`eval_f1_macro`): **run-b**, copiado para `finetuning/models/best/` pelo
próprio `run_experiments.py`.

Leitura honesta: as 3 configurações ficam dentro de ~2pp de f1_macro umas
das outras -- a tarefa é fácil o suficiente (sentença curta, vocabulário
financeiro limitado) para o LoRA convergir bem mesmo com hiperparâmetros
não otimizados. O ganho de run-a → run-b (mais epochs, r maior) é real mas
modesto; run-c (r=16) não supera run-b apesar de mais capacidade,
sugerindo que r=8 já não é o gargalo nesta tarefa.

## 2. Comparação medida: fine-tuned vs. LLM genérico (Groq)

Amostra: 60 frases (20 por classe, estratificada, seed=42) do split de
teste acima -- a mesma amostra para os dois modelos.

| | fine-tuned (LoRA distilbert) | LLM genérico (Groq, `openai/gpt-oss-120b`) |
|---|---|---|
| Accuracy | **0.9833** | 0.9000 |
| Latência média por classificação | **10.7 ms** (GPU local) | 545.2 ms (API, rede incl.) |
| Tokens médios por chamada (in/out) | n/a (não é LLM) | 149.1 / 136.1 |
| Custo real desta corrida | ~0 USD (hardware já amortizado) | 0.00 USD (free tier) |
| Custo calculado à tarifa paga | n/a | **0.1040 USD / 1000** classificações |

Custo à tarifa paga = calculado com os tokens médios REAIS medidos acima ×
o preço público "on-demand" da Groq para `openai/gpt-oss-120b`
($0.15 / M tokens de input, $0.60 / M tokens de output — confirmado em
https://console.groq.com/docs/model/openai/gpt-oss-120b em 2026-08-30).
Não encontrei uma forma fiável de confirmar automaticamente se este preço
mudou desde essa data -- confirma na consola da Groq antes de usar este
número para uma decisão real.

Leitura honesta: nesta tarefa estreita (classificação de sentimento de uma
frase), o modelo fine-tuned é mais preciso (98.3% vs 90.0%) E ~51x mais
rápido, correndo localmente sem custo por chamada. Isto não significa que
o LLM genérico seja pior no projeto todo -- ele continua a fazer coisas
que o classificador não faz (decidir que ferramenta usar, lidar com erros
de forma flexível, o papel de validação do Orquestrador). A troca correta
aqui é: usar o modelo pequeno para a sub-tarefa estreita e repetitiva
(classificar sentimento por título), e manter o LLM para as partes que
exigem raciocínio/flexibilidade -- exatamente o que a integração
`SENTIMENT_PROVIDER=finetuned` faz (ver README raiz).

## 3. Serving (FastAPI + Docker)

Testado localmente (`uvicorn`, GPU) e em container (`docker build`, CPU
-- ver `finetuning/serving/Dockerfile`):

- Endpoint `/classify`, CPU, dentro do container: **45.3 ms** por
  classificação (medição única de smoke test, não é uma média estatística
  -- ver secção 2 para latência média sobre 60 amostras, embora essa tenha
  sido medida em GPU local, não no container).
- Imagem Docker final: **1.72 GB** (`python:3.12-slim` + torch CPU +
  transformers + peft + fastapi/uvicorn).

## 4. Model Registry e monitorização de drift

Medido a 2026-10-25 (`python -m finetuning.registry register-runs` e
`python -m finetuning.drift`, GPU local, versão `champion` = v2).

### Gate de promoção (3 runs registados por ordem cronológica)

| Versão | Run | eval_f1_macro | Decisão | Alias |
|---|---|---|---|---|
| v1 | run-a_lr1e-3_r4_ep3 | 0.9349 | promote (sem campeão) | `previous` |
| v2 | run-b_lr5e-4_r8_ep5 | 0.9564 | promote (+0.0215 sobre v1) | `champion` |
| v3 | run-c_lr2e-4_r16_ep5 | 0.9535 | keep_champion (abaixo de v2) | `challenger` |

### Drift: referência = 500 frases do test split do PhraseBank

| Feature | Referência (média) | Twitter (média) | Controlo: p-value | Twitter: p-value |
|---|---|---|---|---|
| char_length | 124.8 | 86.3 | 0.51 | 3.4e-24 |
| word_count | 23.0 | 12.3 | 0.67 | 4.6e-61 |
| tokens_per_word | 1.24 | 2.10 | 0.77 | 8.1e-91 |
| non_alpha_ratio | 0.076 | 0.086 | 0.99 | 9.1e-04 |
| confidence | 0.989 | 0.957 | 0.29 | 3.5e-29 |
| predicted_label | 60% neutral | 73% neutral | 0.59 | 3.9e-09 |

Testes escolhidos pelo Evidently: K-S (numéricas) e qui-quadrado
(categórica), limite p < 0.05.

| Janela | n | drift_share | f1_macro | Queda vs. campeão | Retreino |
|---|---|---|---|---|---|
| Controlo (outra metade do PhraseBank) | 500 | 0% | 0.966 | -0.010 | não |
| Twitter Financial News (validação) | 500 | 100% | 0.649 | 0.308 | sim |
| Log do serving (300 tweets + 1 frase via `/classify`) | 301 | 100% | sem labels | - | sim |

Leituras:

- O controlo não dispara nada: os limites não dão falsos alarmes na
  mesma distribuição.
- Nos tweets, o f1 cai 31 pontos (accuracy 0.74; f1 por classe: negative
  0.61, neutral 0.82, positive 0.51), mas a confiança média só desce de
  0.989 para 0.957 (respostas com confiança < 0.9: 3% -> 13%). O modelo
  erra com confiança alta, por isso a confiança sozinha não serve de
  proxy da qualidade. O sinal mais forte é `tokens_per_word` (1.24 ->
  2.10): tickers, URLs e calão partem-se em muitos sub-tokens.
- Consequência prática: o modelo só deve ser usado em títulos/frases de
  notícias (o caso do News Analyst). Para tweets seria preciso retreinar
  com dados desse domínio.
