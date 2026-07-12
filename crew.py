"""
Montagem da Crew - Bloco 4.

Junta os 3 agentes (Bloco 3) e as respetivas tarefas (tasks.py) numa Crew
com execucao sequencial: Analista de Noticias -> Analista Tecnico ->
Orquestrador/Critico. A ordem importa porque o Orquestrador precisa dos
outputs dos outros dois como contexto (ver `context=context_tasks` em
`build_validation_task`).
"""

import json
import logging
import re
import time

from crewai import Crew, Process
from pydantic import ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from agents import (
    build_news_analyst_agent,
    build_orchestrator_agent,
    build_technical_analyst_agent,
)
from cache import DEFAULT_TTL_SECONDS, get_cached_report, set_cached_report
from metrics import record_analysis
from rate_limiter import (
    ESTIMATED_TOKENS_PER_ANALYSIS,
    ESTIMATED_TOKENS_PER_CALL,
    groq_rate_limiter,
)
from schemas import RelatorioFinanceiro
from tasks import build_news_task, build_technical_task, build_validation_task

logger = logging.getLogger(__name__)


def is_rate_limit_error(exc: BaseException) -> bool:
    """
    Deteta se uma excecao (ou qualquer causa na sua cadeia __cause__) e um
    rate limit da Groq.

    Nao podemos confiar apenas no tipo da excecao: quando o CrewAI usa
    `output_pydantic`, o erro da Groq chega envolvido em camadas por
    `instructor` (InstructorRetryException) e por vezes `litellm`, e o tipo
    exato varia consoante o caminho de codigo (chamada normal vs chamada
    com output_pydantic). Verificar a mensagem ao longo de toda a cadeia de
    causas e mais robusto do que apostar num tipo especifico.
    """
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if (
            "rate_limit_exceeded" in message
            or "rate limit" in message
            or "429" in message
        ):
            return True
        current = current.__cause__
    return False


def _extract_json_report(raw_text: str) -> RelatorioFinanceiro | None:
    """
    Extrai e valida o RelatorioFinanceiro a partir do texto em bruto devolvido
    pelo Orquestrador.

    Fazemos isto manualmente (em vez de usar `output_pydantic`, que forca o
    CrewAI a exigir uma tool-call estruturada) porque forcar tool-calling em
    modelos Groq como o gpt-oss-120b e instavel: o modelo por vezes devolve o
    JSON correto como texto simples em vez de uma chamada de tool formal, e a
    API da Groq rejeita isso com um erro 400 ("Tool choice is required, but
    model did not call a tool") mesmo quando o conteudo gerado esta correto e
    visivel no campo `failed_generation` do erro. Validar o JSON manualmente
    aqui evita esse problema por completo, e ainda poupa uma chamada extra ao
    LLM (a que o `instructor` faria para forcar o schema).

    Returns:
        A instancia validada, ou None se o texto nao contiver JSON valido
        que respeite o schema.
    """
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
        return RelatorioFinanceiro.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Falha ao fazer parse/validar o JSON do Orquestrador: %s", exc)
        return None


def build_crew(stage_callback=None) -> Crew:
    """
    Constroi a Crew completa: 3 agentes + 3 tarefas em processo sequencial.

    Args:
        stage_callback: funcao opcional `(stage: str) -> None` chamada
            sempre que a analise avanca para uma nova fase do pipeline
            ("tecnico" quando a tarefa de noticias termina, "orquestrador"
            quando a tarefa tecnica termina). Usado pelo frontend para
            mostrar progresso real em vez de um spinner generico.

    Returns:
        Instancia de `crewai.Crew` pronta a ser executada com `.kickoff(inputs=...)`.
    """
    news_agent = build_news_analyst_agent()
    technical_agent = build_technical_analyst_agent()
    orchestrator_agent = build_orchestrator_agent()

    news_task = build_news_task(news_agent, on_stage_change=stage_callback)
    technical_task = build_technical_task(
        technical_agent, on_stage_change=stage_callback
    )
    validation_task = build_validation_task(
        orchestrator_agent, context_tasks=[news_task, technical_task]
    )

    return Crew(
        agents=[news_agent, technical_agent, orchestrator_agent],
        tasks=[news_task, technical_task, validation_task],
        process=Process.sequential,
        verbose=True,
    )


@retry(
    retry=retry_if_exception(is_rate_limit_error),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=15, max=90),
    reraise=True,
)
def _kickoff_with_retry(crew: Crew, inputs: dict):
    """
    Wrapper com retry para o rate limit gratuito da Groq (8000 tokens/minuto
    no tier 'on_demand'). Esperamos com backoff exponencial (15s, 30s, 60s,
    90s) em vez de falhar imediatamente -- necessario mesmo com as pausas
    intra-execucao em tasks.py, como rede de seguranca adicional.
    """
    return crew.kickoff(inputs=inputs)


def run_analysis(
    asset_name: str,
    ticker: str,
    stage_callback=None,
    use_cache: bool = True,
    cache_ttl_seconds: int = DEFAULT_TTL_SECONDS,
):
    """
    Executa a analise completa para um ativo especifico.

    Args:
        asset_name: nome do ativo em linguagem natural (ex: "Tesla").
        ticker: simbolo do ativo (ex: "TSLA").
        stage_callback: funcao opcional `(stage: str) -> None` chamada em
            cada transicao de fase do pipeline: "noticias" (inicio),
            "tecnico", "orquestrador", "concluido". Ver `build_crew`.
        use_cache: se True (default), verifica primeiro a cache SQLite
            (ver cache.py) e devolve o resultado em cache sem chamar a
            Groq se ainda estiver dentro do TTL. Usa `use_cache=False`
            para forcar uma analise fresca.
        cache_ttl_seconds: por quanto tempo um resultado em cache e
            considerado valido antes de ser re-analisado.

    Returns:
        Uma instancia de `schemas.RelatorioFinanceiro` se conseguirmos
        extrair e validar JSON valido a partir do output do Orquestrador
        (ou se veio da cache). Se a extracao falhar (ex: o LLM devolveu
        algo irrecuperavel), devolve o texto em bruto (str) para nao
        perder o resultado -- quem chama esta funcao deve verificar o
        tipo devolvido (ver main.py para o padrao recomendado).

    Raises:
        Exception: se o rate limit da Groq persistir mesmo apos os retries
            com backoff (ver `_kickoff_with_retry`) -- o tipo exato depende
            do caminho interno do CrewAI/litellm, por isso quem chama esta
            funcao deve verificar a mensagem em vez de um tipo especifico
            (ver `is_rate_limit_error` para o mesmo criterio usado aqui).
    """
    if use_cache:
        cached = get_cached_report(ticker, ttl_seconds=cache_ttl_seconds)
        if cached is not None:
            logger.info(
                "Cache hit para %s (%s) -- a saltar a chamada à Groq.",
                asset_name,
                ticker,
            )
            if stage_callback is not None:
                stage_callback("concluido")
            record_analysis(asset_name, ticker, 0.0, status="cache_hit", tokens=None)
            return cached

    if stage_callback is not None:
        stage_callback("noticias")

    # Reserva orcamento no token bucket para a primeira chamada ao LLM
    # (as chamadas seguintes sao geridas pelos callbacks das tasks -- ver
    # tasks.py). Substitui o antigo `time.sleep` fixo entre analises.
    waited = groq_rate_limiter.acquire(ESTIMATED_TOKENS_PER_CALL)
    if waited > 0:
        logger.info(
            "Rate limiter: à espera de %.1fs antes de iniciar a análise de %s (orçamento Groq).",
            waited,
            asset_name,
        )

    start_time = time.monotonic()
    crew = build_crew(stage_callback=stage_callback)
    logger.info("A iniciar analise para asset_name=%s, ticker=%s", asset_name, ticker)

    try:
        crew_output = _kickoff_with_retry(
            crew, {"asset_name": asset_name, "ticker": ticker}
        )
    except Exception as exc:
        duration = time.monotonic() - start_time
        record_analysis(
            asset_name,
            ticker,
            duration,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    duration = time.monotonic() - start_time

    if stage_callback is not None:
        stage_callback("concluido")

    tokens = None
    usage = getattr(crew_output, "token_usage", None)
    if usage is not None:
        tokens = {
            "total": getattr(usage, "total_tokens", None),
            "prompt": getattr(usage, "prompt_tokens", None),
            "completion": getattr(usage, "completion_tokens", None),
            "successful_requests": getattr(usage, "successful_requests", None),
        }
        # Corrige o token bucket com o consumo REAL desta analise (3
        # chamadas ao LLM), para nao ir derivando da realidade ao longo de
        # muitas execucoes -- ver rate_limiter.TokenBucket.reconcile.
        if tokens["total"] is not None:
            groq_rate_limiter.reconcile(tokens["total"], ESTIMATED_TOKENS_PER_ANALYSIS)

    report = _extract_json_report(crew_output.raw)
    if report is not None:
        record_analysis(asset_name, ticker, duration, status="success", tokens=tokens)
        if use_cache:
            set_cached_report(ticker, asset_name, report)
        return report

    logger.warning(
        "Não foi possível extrair um RelatorioFinanceiro válido do output do "
        "Orquestrador. A devolver o texto em bruto -- ver main.py para como "
        "isto é tratado (status 'invalid_schema')."
    )
    record_analysis(
        asset_name, ticker, duration, status="invalid_schema", tokens=tokens
    )
    return crew_output.raw


if __name__ == "__main__":
    # Teste manual: confirma que a Crew e montada sem erros estruturais.
    # NAO executa kickoff() aqui de proposito -- isso faz chamadas reais ao
    # LLM e as ferramentas, o que consome quota da API Groq. A execucao real
    # e feita explicitamente pelo utilizador (ver README, secao Bloco 4).
    print("--- Teste de montagem da Crew (Bloco 4) ---")
    try:
        crew = build_crew()
        print(
            f"Crew montada com {len(crew.agents)} agentes e {len(crew.tasks)} tarefas."
        )
        for i, task in enumerate(crew.tasks, start=1):
            print(f"  Tarefa {i}: agente='{task.agent.role}'")
        print("\nCrew montada com sucesso. Para executar uma analise real, usa:")
        print("  from crew import run_analysis")
        print('  run_analysis(asset_name="Tesla", ticker="TSLA")')
    except Exception as exc:
        print(f"Falha ao montar a Crew: {type(exc).__name__}: {exc}")
