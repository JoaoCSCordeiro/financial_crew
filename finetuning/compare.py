"""
Compara o modelo fine-tuned (LoRA sobre distilbert-base-uncased) com a
chamada LLM genérica atual do News Analyst (Groq, `openai/gpt-oss-120b`)
na mesma tarefa: classificação de sentimento de uma frase financeira em
positivo/negativo/neutro.

Tudo o que este script reporta é MEDIDO na hora (accuracy, latência,
tokens reais gastos), exceto o preço público por token da Groq (tarifa
"on-demand", citada com fonte e data -- ver `GROQ_ON_DEMAND_PRICE_USD_PER_M`
abaixo), que é multiplicado pelos tokens REAIS medidos para dar um custo
por 1000 classificações -- não é um número inventado, é uma conta feita
com um preço público verificável e consumo medido.

Requer GROQ_API_KEY real no .env (usa a API da Groq diretamente via REST,
não via crewai/litellm -- ver DECISOES-FINETUNING.md, decisão sobre a
incompatibilidade litellm/Python 3.10 encontrada neste ambiente).

Uso:
    python -m finetuning.compare --n-per-class 20
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv

from finetuning.constants import LABEL_NAMES
from finetuning.data import load_test_examples
from finetuning.inference import SentimentClassifier
from finetuning.tracking import BEST_MODEL_DIR
from rate_limiter import TokenBucket

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_ESTIMATED_TOKENS_PER_CALL = (
    250  # chamada curta, single-sentence -- ver acquire() abaixo
)

# Preço público "on-demand" da Groq para openai/gpt-oss-120b, confirmado em
# https://console.groq.com/docs/model/openai/gpt-oss-120b em 2026-08-30.
# NÃO é o custo real desta comparação (que corre no free tier, custo real
# = 0 USD) -- é usado só para calcular quanto custaria à tarifa paga,
# aplicado aos tokens REAIS medidos nesta corrida.
GROQ_ON_DEMAND_PRICE_USD_PER_M_INPUT = 0.15
GROQ_ON_DEMAND_PRICE_USD_PER_M_OUTPUT = 0.60

CLASSIFICATION_PROMPT = (
    "Classifica o sentimento da seguinte frase financeira para o preço/perspetivas "
    "do ativo mencionado. Responde APENAS com uma destas 3 palavras, sem mais "
    "nenhum texto: positive, negative, ou neutral.\n\nFrase: {text}"
)


@dataclass
class ModelBenchmark:
    name: str
    n_examples: int = 0
    correct: int = 0
    latencies_ms: list = field(default_factory=list)
    prompt_tokens: list = field(default_factory=list)
    completion_tokens: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n_examples if self.n_examples else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return (
            sum(self.latencies_ms) / len(self.latencies_ms)
            if self.latencies_ms
            else 0.0
        )

    @property
    def avg_prompt_tokens(self) -> float:
        return (
            sum(self.prompt_tokens) / len(self.prompt_tokens)
            if self.prompt_tokens
            else 0.0
        )

    @property
    def avg_completion_tokens(self) -> float:
        return (
            sum(self.completion_tokens) / len(self.completion_tokens)
            if self.completion_tokens
            else 0.0
        )


def _stratified_sample(n_per_class: int, seed: int = 42) -> list:
    examples = load_test_examples()
    by_label: dict[str, list] = {name: [] for name in LABEL_NAMES}
    for ex in examples:
        by_label[ex.label_text].append(ex)

    rng = random.Random(seed)
    sample = []
    for name in LABEL_NAMES:
        pool = by_label[name]
        sample.extend(rng.sample(pool, min(n_per_class, len(pool))))
    rng.shuffle(sample)
    return sample


def _parse_groq_label(content: str) -> str | None:
    content = content.strip().lower()
    for name in LABEL_NAMES:
        if name in content:
            return name
    return None


def benchmark_finetuned(examples: list) -> ModelBenchmark:
    if not BEST_MODEL_DIR.exists():
        raise RuntimeError(
            f"Nenhum modelo em {BEST_MODEL_DIR}. Corre "
            "`python -m finetuning.run_experiments` primeiro."
        )
    classifier = SentimentClassifier(BEST_MODEL_DIR)
    bench = ModelBenchmark(name="fine-tuned (LoRA distilbert-base-uncased)")

    # Aquecimento -- exclui o custo de inicialização (kernels CUDA, cache) da medição.
    classifier.predict(examples[0].text)

    for ex in examples:
        start = time.perf_counter()
        result = classifier.predict(ex.text)
        elapsed_ms = (time.perf_counter() - start) * 1000

        bench.n_examples += 1
        bench.latencies_ms.append(elapsed_ms)
        if result["label"] == ex.label_text:
            bench.correct += 1

    return bench


def _parse_reset_seconds(header_value: str | None, default: float = 5.0) -> float:
    """Converte o header `x-ratelimit-reset-tokens` da Groq (ex: '37.92s', '1m5s') em segundos."""
    if not header_value:
        return default
    header_value = header_value.strip()
    total = 0.0
    number = ""
    for char in header_value:
        if char.isdigit() or char == ".":
            number += char
        elif char == "m" and number:
            total += float(number) * 60
            number = ""
        elif char == "s" and number:
            total += float(number)
            number = ""
    return total if total > 0 else default


def _post_groq_with_retry(
    api_key: str, model_name: str, prompt: str, max_attempts: int = 5
) -> tuple[dict, float]:
    """POST à API da Groq, com retry automático quando o próprio 429 indica o tempo de espera."""
    for attempt in range(1, max_attempts + 1):
        start = time.perf_counter()
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=30,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if response.status_code == 429 and attempt < max_attempts:
            wait_seconds = _parse_reset_seconds(
                response.headers.get("retry-after")
                or response.headers.get("x-ratelimit-reset-tokens")
            )
            logging.getLogger(__name__).warning(
                "Groq 429 (tentativa %d/%d) -- à espera de %.1fs (x-ratelimit-reset-tokens).",
                attempt,
                max_attempts,
                wait_seconds,
            )
            time.sleep(wait_seconds + 0.5)
            continue

        response.raise_for_status()
        return response.json(), elapsed_ms

    raise RuntimeError("Groq: excedidas as tentativas de retry após 429 repetidos.")


def benchmark_groq(examples: list, model_name: str) -> ModelBenchmark:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "cola_aqui_a_tua_chave_groq":
        raise RuntimeError(
            "GROQ_API_KEY não configurada -- necessária para medir a comparação real "
            "com o LLM (ver .env)."
        )

    bucket = TokenBucket(capacity=8000, refill_period_seconds=60.0)
    bench = ModelBenchmark(name=f"LLM genérico (Groq, {model_name})")

    for ex in examples:
        bucket.acquire(GROQ_ESTIMATED_TOKENS_PER_CALL)

        prompt = CLASSIFICATION_PROMPT.format(text=ex.text)
        payload, elapsed_ms = _post_groq_with_retry(api_key, model_name, prompt)

        usage = payload.get("usage", {})
        actual_tokens = usage.get("total_tokens", GROQ_ESTIMATED_TOKENS_PER_CALL)
        bucket.reconcile(actual_tokens, GROQ_ESTIMATED_TOKENS_PER_CALL)

        content = payload["choices"][0]["message"]["content"]
        predicted = _parse_groq_label(content)

        bench.n_examples += 1
        bench.latencies_ms.append(elapsed_ms)
        bench.prompt_tokens.append(usage.get("prompt_tokens", 0))
        bench.completion_tokens.append(usage.get("completion_tokens", 0))
        if predicted == ex.label_text:
            bench.correct += 1

    return bench


def cost_per_1000_usd(bench: ModelBenchmark) -> float:
    input_cost = (
        bench.avg_prompt_tokens * GROQ_ON_DEMAND_PRICE_USD_PER_M_INPUT / 1_000_000
    )
    output_cost = (
        bench.avg_completion_tokens * GROQ_ON_DEMAND_PRICE_USD_PER_M_OUTPUT / 1_000_000
    )
    return (input_cost + output_cost) * 1000


def print_report(finetuned: ModelBenchmark, groq: ModelBenchmark) -> None:
    print("\n" + "=" * 70)
    print("COMPARAÇÃO MEDIDA: fine-tuned vs. LLM genérico (Groq)")
    print("=" * 70)
    for bench in (finetuned, groq):
        print(f"\n{bench.name}")
        print(f"  N amostras:            {bench.n_examples}")
        print(f"  Accuracy:              {bench.accuracy:.4f}")
        print(f"  Latência média:        {bench.avg_latency_ms:.1f} ms")
        if bench.prompt_tokens:
            print(
                f"  Tokens médios (in/out): {bench.avg_prompt_tokens:.1f} / {bench.avg_completion_tokens:.1f}"
            )
            print("  Custo real (free tier): 0.00 USD / 1000 classificações")
            print(
                f"  Custo calculado à tarifa on-demand publicada "
                f"(${GROQ_ON_DEMAND_PRICE_USD_PER_M_INPUT}/{GROQ_ON_DEMAND_PRICE_USD_PER_M_OUTPUT} "
                f"por M tokens in/out): {cost_per_1000_usd(bench):.4f} USD / 1000 classificações"
            )
        else:
            print(
                "  Custo:                 ~0 USD / 1000 (inferência local, hardware já amortizado)"
            )
    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--groq-model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    )
    args = parser.parse_args()

    examples = _stratified_sample(args.n_per_class, seed=args.seed)
    print(
        f"Amostra estratificada: {len(examples)} exemplos "
        f"({args.n_per_class} por classe) do split de teste do Financial PhraseBank."
    )

    print("\nA correr o modelo fine-tuned...")
    finetuned_bench = benchmark_finetuned(examples)

    print(f"A correr o LLM genérico (Groq, {args.groq_model})...")
    groq_bench = benchmark_groq(examples, args.groq_model)

    print_report(finetuned_bench, groq_bench)


if __name__ == "__main__":
    main()
