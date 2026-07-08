"""
Montagem da Crew - Bloco 4.

Junta os 3 agentes (Bloco 3) e as respetivas tarefas (tasks.py) numa Crew
com execucao sequencial: Analista de Noticias -> Analista Tecnico ->
Orquestrador/Critico. A ordem importa porque o Orquestrador precisa dos
outputs dos outros dois como contexto (ver `context=context_tasks` em
`build_validation_task`).
"""

import logging

from crewai import Crew, Process
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from agents import (
    build_news_analyst_agent,
    build_orchestrator_agent,
    build_technical_analyst_agent,
)
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
        if "rate_limit_exceeded" in message or "rate limit" in message or "429" in message:
            return True
        current = current.__cause__
    return False


def build_crew() -> Crew:
    """
    Constroi a Crew completa: 3 agentes + 3 tarefas em processo sequencial.

    Returns:
        Instancia de `crewai.Crew` pronta a ser executada com `.kickoff(inputs=...)`.
    """
    news_agent = build_news_analyst_agent()
    technical_agent = build_technical_analyst_agent()
    orchestrator_agent = build_orchestrator_agent()

    news_task = build_news_task(news_agent)
    technical_task = build_technical_task(technical_agent)
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
    no tier 'on_demand'). Um unico kickoff desta Crew pode facilmente
    consumir varios milhares de tokens (3 chamadas ao LLM + a chamada extra
    do 'instructor' para validar o output_pydantic), por isso e normal
    esbarrar no limite ao correr varios ativos seguidos. Esperamos com
    backoff exponencial (15s, 30s, 60s, 90s) em vez de falhar imediatamente.
    """
    return crew.kickoff(inputs=inputs)


def run_analysis(asset_name: str, ticker: str):
    """
    Executa a analise completa para um ativo especifico.

    Args:
        asset_name: nome do ativo em linguagem natural (ex: "Tesla").
        ticker: simbolo do ativo (ex: "TSLA").

    Returns:
        Uma instancia de `schemas.RelatorioFinanceiro` se o CrewAI conseguiu
        validar o output do Orquestrador contra o schema Pydantic (caminho
        esperado). Se, por alguma razao, a validacao Pydantic falhar (ex: o
        LLM devolveu algo irrecuperavel), devolve o output em bruto (string
        ou dict) para nao perder o resultado -- quem chama esta funcao deve
        verificar o tipo devolvido (ver main.py para o padrao recomendado).

    Raises:
        Exception: se o rate limit da Groq persistir mesmo apos os retries
            com backoff (ver `_kickoff_with_retry`) -- o tipo exato depende
            do caminho interno do CrewAI/instructor, por isso quem chama
            esta funcao deve verificar a mensagem em vez de um tipo especifico
            (ver `_is_rate_limit_error` para o mesmo criterio usado aqui).
    """
    crew = build_crew()
    logger.info("A iniciar analise para asset_name=%s, ticker=%s", asset_name, ticker)

    crew_output = _kickoff_with_retry(crew, {"asset_name": asset_name, "ticker": ticker})

    if crew_output.pydantic is not None:
        return crew_output.pydantic

    logger.warning(
        "O output do Orquestrador não passou na validação Pydantic estrita "
        "(RelatorioFinanceiro). A devolver o output em bruto -- ver "
        "'notas_de_validacao' não estará disponível neste caso."
    )
    return crew_output.json_dict or crew_output.raw


if __name__ == "__main__":
    # Teste manual: confirma que a Crew e montada sem erros estruturais.
    # NAO executa kickoff() aqui de proposito -- isso faz chamadas reais ao
    # LLM e as ferramentas, o que consome quota da API Groq. A execucao real
    # e feita explicitamente pelo utilizador (ver README, secao Bloco 4).
    print("--- Teste de montagem da Crew (Bloco 4) ---")
    try:
        crew = build_crew()
        print(f"Crew montada com {len(crew.agents)} agentes e {len(crew.tasks)} tarefas.")
        for i, task in enumerate(crew.tasks, start=1):
            print(f"  Tarefa {i}: agente='{task.agent.role}'")
        print("\nCrew montada com sucesso. Para executar uma analise real, usa:")
        print('  from crew import run_analysis')
        print('  run_analysis(asset_name="Tesla", ticker="TSLA")')
    except Exception as exc:
        print(f"Falha ao montar a Crew: {type(exc).__name__}: {exc}")
