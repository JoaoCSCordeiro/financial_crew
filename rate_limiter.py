"""
Rate Limiter — Token Bucket para o orçamento gratuito da Groq.

Substitui os `time.sleep(N)` fixos (usados nas versões anteriores deste
projeto) por um token bucket real: a espera é exatamente o tempo
necessário para o orçamento reabastecer (frequentemente zero), não um
atraso cego. Mais rápido no caso comum, e continua seguro quando o
orçamento está apertado.

O bucket é um singleton a nível de processo -- múltiplas análises
sequenciais (o loop do `main.py`, ou o worker do frontend) partilham o
MESMO orçamento de 8000 tokens/minuto da Groq, em vez de cada uma achar
que tem a quota toda para si. Nota: por ser em memória, o orçamento não é
partilhado entre processos diferentes (ex: `main.py` e `frontend/app.py`
corridos ao mesmo tempo) -- para isso seria necessário um bucket externo
(Redis, ficheiro partilhado com lock), fora do âmbito deste projeto.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class TokenBucket:
    """Implementação simples e thread-safe do algoritmo de token bucket."""

    def __init__(self, capacity: float, refill_period_seconds: float = 60.0):
        self.capacity = capacity
        self.refill_period_seconds = refill_period_seconds
        self.refill_rate = capacity / refill_period_seconds  # tokens por segundo
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        """Reabastece o bucket com base no tempo decorrido. Chamar só com o lock detido."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
            self._last_refill = now

    def acquire(self, tokens: float) -> float:
        """
        Bloqueia até `tokens` estarem disponíveis, depois deduz-nos do bucket.

        Args:
            tokens: quantidade de tokens a reservar (nunca superior à
                capacidade total do bucket -- é automaticamente limitado).

        Returns:
            Quantos segundos esperou (0.0 se já havia orçamento suficiente).
            Útil para logging/observabilidade.
        """
        tokens = min(tokens, self.capacity)

        with self._lock:
            self._refill_locked()
            deficit = tokens - self._tokens
            wait_seconds = max(0.0, deficit / self.refill_rate) if deficit > 0 else 0.0

        if wait_seconds > 0:
            time.sleep(wait_seconds)
            with self._lock:
                self._refill_locked()

        with self._lock:
            self._tokens = max(0.0, self._tokens - tokens)

        return wait_seconds

    def reconcile(self, actual_tokens: float, estimated_tokens: float) -> None:
        """
        Ajusta o estado do bucket depois de sabermos o consumo REAL (via
        `CrewOutput.token_usage`), corrigindo o desvio entre a estimativa
        usada em `acquire()` e a realidade -- evita que o bucket vá
        "derivando" ao longo de muitas execuções.
        """
        diff = estimated_tokens - actual_tokens  # positivo se sobrestimámos
        with self._lock:
            self._refill_locked()
            self._tokens = max(0.0, min(self.capacity, self._tokens + diff))

    def available(self) -> float:
        """Tokens atualmente disponíveis (após reabastecimento), para inspeção/debug."""
        with self._lock:
            self._refill_locked()
            return self._tokens


# Estimativa conservadora de tokens por chamada ao LLM (prompt + completion),
# calibrada com base nas execuções reais observadas durante o desenvolvimento
# (Blocos 4/5 -- ver README, "Groq free tier rate limits"). É corrigida após
# cada análise via `reconcile()`, usando o consumo real devolvido pela Groq.
ESTIMATED_TOKENS_PER_CALL = 1800

# Uma análise completa faz 3 chamadas ao LLM (notícias, técnico, orquestrador).
ESTIMATED_TOKENS_PER_ANALYSIS = 3 * ESTIMATED_TOKENS_PER_CALL

# Singleton partilhado por todo o processo.
groq_rate_limiter = TokenBucket(capacity=8000, refill_period_seconds=60.0)
