"""
Definicoes das Tarefas (Tasks) - Bloco 4.

Cada Task esta ligada a um Agente especifico do Bloco 3. Os placeholders
{asset_name} e {ticker} sao preenchidos em tempo de execucao pelo CrewAI
a partir do dicionario `inputs` passado a `crew.kickoff()` (ver crew.py),
permitindo reutilizar a mesma Crew para qualquer ativo sem reescrever nada.

O `expected_output` de cada tarefa e deliberadamente especifico -- isto
nao e apenas documentacao, e parte do prompt que o LLM recebe, e ajuda a
reduzir divergencia de formato entre execucoes.
"""

import time

from crewai import Agent, Task

# Pausa entre tarefas do MESMO ativo, para respeitar o rate limit gratuito
# da Groq (8000 tokens/minuto no modelo openai/gpt-oss-120b). Uma analise
# completa faz 4 chamadas ao LLM (noticias, tecnico, orquestrador + a
# chamada extra do 'instructor' para validar o schema) muito seguidas --
# isso sozinho pode esgotar o limite antes mesmo de passarmos para o
# proximo ativo. Espacar as tarefas dentro do mesmo ativo da tempo ao
# limite (janela rolante por minuto) para libertar capacidade.
SECONDS_BETWEEN_TASKS = 25


def _make_task_callback(on_stage_change=None, next_stage: str | None = None):
    """
    Constroi o callback executado quando uma tarefa termina: espaca a
    proxima chamada ao LLM (rate limit da Groq) e, opcionalmente, avisa
    quem estiver a ouvir (ex: o frontend) de que a analise avancou para a
    proxima fase do pipeline.
    """

    def _callback(task_output) -> None:
        if on_stage_change is not None and next_stage is not None:
            on_stage_change(next_stage)
        time.sleep(SECONDS_BETWEEN_TASKS)

    return _callback


def build_news_task(agent: Agent, on_stage_change=None) -> Task:
    """Tarefa do Agente 1: recolher e resumir noticias recentes do ativo."""
    return Task(
        description=(
            "Usa a ferramenta 'obter_noticias_recentes' com max_results=5 para "
            "pesquisar noticias sobre '{asset_name}'. Classifica CADA um dos "
            "titulos devolvidos individualmente como positivo, negativo ou "
            "neutro para o preco/perspetivas do ativo, e so depois conta os "
            "resultados para decidir o sentimento geral. Regra de decisao "
            "obrigatoria (aplica-a literalmente, nao uses a tua impressao "
            "geral): sejam P = numero de titulos positivos e N = numero de "
            "titulos negativos (ignora os neutros nesta conta):\n"
            "- Se P > N: sentimento geral = 'positivo'\n"
            "- Se N > P: sentimento geral = 'negativo'\n"
            "- Se P == N e ambos >= 1: sentimento geral = 'misto'\n"
            "- Se P == N == 0 (todos neutros): sentimento geral = 'neutro'\n"
            "NAO uses 'misto' como resposta por defeito quando estiveres "
            "inseguro -- e um resultado especifico que so se aplica quando "
            "P e N sao realmente iguais e ambos maiores que zero. Mostra a "
            "tua contagem (quantos titulos positivos, negativos, neutros) "
            "antes da conclusao final, para que seja possivel auditar a "
            "decisao. Se a ferramenta devolver um erro ou zero noticias, "
            "reporta isso explicitamente em vez de inventar contexto."
        ),
        expected_output=(
            "Um resumo estruturado em texto contendo: (1) a classificacao "
            "individual de cada titulo (positivo/negativo/neutro) com uma "
            "justificacao de uma frase, (2) a contagem final (P=x, N=y, "
            "neutros=z), (3) o sentimento geral resultante da aplicacao "
            "literal da regra de decisao, (4) a lista dos titulos usados "
            "como evidencia (com fonte e data, exatamente como devolvidos "
            "pela ferramenta), e (5) um campo explicito indicando se a "
            "pesquisa falhou ou nao devolveu resultados."
        ),
        agent=agent,
        callback=_make_task_callback(on_stage_change, next_stage="tecnico"),
    )


def build_technical_task(agent: Agent, on_stage_change=None) -> Task:
    """Tarefa do Agente 2: obter e interpretar dados tecnicos do ativo."""
    return Task(
        description=(
            "Usa a ferramenta 'obter_dados_mercado' com o ticker '{ticker}' para "
            "obter o preco atual e o historico de 1 mes. Com base APENAS nesses "
            "dados, produz uma analise tecnica objetiva: tendencia (alta/baixa/"
            "lateral), variacao percentual no periodo, e amplitude entre maximo "
            "e minimo. Se a ferramenta devolver um erro (ticker invalido, falha "
            "de rede), reporta o erro tal como foi devolvido."
        ),
        expected_output=(
            "Um resumo estruturado em texto contendo: (1) o preco atual e a "
            "moeda, (2) a variacao percentual no ultimo mes, (3) a tendencia "
            "identificada, (4) os valores de maximo/minimo/media do periodo, "
            "todos exatamente como devolvidos pela ferramenta, e (5) um campo "
            "explicito indicando se a obtencao de dados falhou."
        ),
        agent=agent,
        callback=_make_task_callback(on_stage_change, next_stage="orquestrador"),
    )


def build_validation_task(agent: Agent, context_tasks: list[Task]) -> Task:
    """
    Tarefa do Agente 3: validar e consolidar os outputs das duas tarefas
    anteriores num unico relatorio JSON.

    `context_tasks` liga esta tarefa aos outputs das tarefas do Agente 1 e
    do Agente 2 -- o CrewAI injeta automaticamente esses resultados no
    contexto desta tarefa, para o Orquestrador poder valida-los.
    """
    return Task(
        description=(
            "Revê cuidadosamente os outputs do Analista de Noticias e do "
            "Analista Tecnico para o ativo '{asset_name}' ({ticker}). "
            "Verifica que cada numero e cada titulo de noticia no relatorio "
            "final tem correspondencia direta com o que os outros dois "
            "agentes reportaram -- rejeita e sinaliza qualquer coisa que "
            "pareca inventada, extrapolada ou nao rastreavel. Consolida tudo "
            "num unico relatorio JSON final, seguindo exatamente a estrutura "
            "pedida em 'expected_output'. Se algum dado estiver em falta ou "
            "invalido, reflete isso nos campos apropriados em vez de omitir "
            "ou inventar um valor plausivel."
        ),
        expected_output=(
            "APENAS um objeto JSON valido (sem markdown, sem texto antes ou "
            "depois), com exatamente esta estrutura:\n"
            "{\n"
            '  "ativo": "<nome do ativo>",\n'
            '  "ticker": "<ticker>",\n'
            '  "dados_tecnicos": {\n'
            '    "preco_atual": <numero ou null>,\n'
            '    "moeda": "<string ou null>",\n'
            '    "variacao_percentual_1_mes": <numero ou null>,\n'
            '    "tendencia": "<alta|baixa|lateral ou null>",\n'
            '    "dados_indisponiveis": <true|false>\n'
            "  },\n"
            '  "analise_noticias": {\n'
            '    "sentimento_geral": "<positivo|negativo|neutro|misto ou null>",\n'
            '    "titulos_usados": ["<titulo 1>", "..."],\n'
            '    "dados_indisponiveis": <true|false>\n'
            "  },\n"
            '  "notas_de_validacao": "<contradicoes ou problemas encontrados, ou string vazia>"\n'
            "}"
        ),
        agent=agent,
        context=context_tasks,
    )
