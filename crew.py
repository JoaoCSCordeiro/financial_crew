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

from agents import (
    build_news_analyst_agent,
    build_orchestrator_agent,
    build_technical_analyst_agent,
)
from tasks import build_news_task, build_technical_task, build_validation_task

logger = logging.getLogger(__name__)


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


def run_analysis(asset_name: str, ticker: str) -> str:
    """
    Executa a analise completa para um ativo especifico.

    Args:
        asset_name: nome do ativo em linguagem natural (ex: "Tesla").
        ticker: simbolo do ativo (ex: "TSLA").

    Returns:
        O output final (esperado: string JSON) produzido pelo Agente 3.
    """
    crew = build_crew()
    logger.info("A iniciar analise para asset_name=%s, ticker=%s", asset_name, ticker)

    result = crew.kickoff(inputs={"asset_name": asset_name, "ticker": ticker})
    return str(result)


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
