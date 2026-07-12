"""Testes para rate_limiter.py -- token bucket."""

import time

from rate_limiter import TokenBucket


def test_bucket_comeca_cheio():
    bucket = TokenBucket(capacity=1000, refill_period_seconds=60.0)
    assert bucket.available() == 1000


def test_acquire_sem_espera_quando_ha_orcamento():
    bucket = TokenBucket(capacity=1000, refill_period_seconds=60.0)
    waited = bucket.acquire(200)
    assert waited == 0.0
    assert (
        abs(bucket.available() - 800) < 1.0
    )  # pequena margem por reabastecimento continuo


def test_acquire_deduz_tokens_corretamente():
    bucket = TokenBucket(capacity=1000, refill_period_seconds=60.0)
    bucket.acquire(300)
    bucket.acquire(300)
    assert bucket.available() < 401  # 1000 - 600, com pequena margem de reabastecimento


def test_acquire_espera_quando_orcamento_insuficiente():
    # Bucket pequeno com refill rapido para o teste ser rapido: 10 tokens/segundo.
    bucket = TokenBucket(capacity=10, refill_period_seconds=1.0)
    bucket.acquire(10)  # esgota o bucket

    start = time.monotonic()
    waited = bucket.acquire(5)  # precisa de 0.5s para reabastecer 5 tokens
    elapsed = time.monotonic() - start

    assert waited > 0
    assert elapsed >= waited * 0.8  # o sleep real deve corresponder ao tempo devolvido


def test_reconcile_ajusta_para_cima_quando_sobrestimado():
    bucket = TokenBucket(capacity=1000, refill_period_seconds=60.0)
    bucket.acquire(500)  # reservamos 500
    # Consumo real foi so 300 -- devolvemos a diferenca (200) ao bucket.
    bucket.reconcile(actual_tokens=300, estimated_tokens=500)
    assert bucket.available() > 500  # deve ter recuperado tokens


def test_reconcile_ajusta_para_baixo_quando_subestimado():
    bucket = TokenBucket(capacity=1000, refill_period_seconds=60.0)
    bucket.acquire(300)  # reservamos so 300
    # Consumo real foi maior (600) -- deduzimos mais 300 do bucket.
    bucket.reconcile(actual_tokens=600, estimated_tokens=300)
    assert bucket.available() < 700  # deve ter deduzido mais


def test_reconcile_nunca_excede_capacidade():
    bucket = TokenBucket(capacity=1000, refill_period_seconds=60.0)
    bucket.reconcile(actual_tokens=0, estimated_tokens=5000)  # diff enorme positivo
    assert bucket.available() <= 1000


def test_acquire_nunca_pede_mais_que_a_capacidade():
    bucket = TokenBucket(capacity=100, refill_period_seconds=60.0)
    # Pedir mais do que a capacidade total não deve bloquear para sempre.
    waited = bucket.acquire(10_000)
    assert waited == 0.0  # foi limitado a capacity=100, que já estava disponível
