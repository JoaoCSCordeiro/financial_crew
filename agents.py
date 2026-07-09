"""
Definições dos Agentes — Bloco 3.

Arquitetura neuro-simbólica: os Agentes 1 e 2 são "sensores" que só podem
reportar o que as ferramentas determinísticas (yfinance/ddgs) devolveram.
O Agente 3 é o componente simbólico/validador — a sua função não é gerar
conteúdo novo, é AUDITAR o que os Agentes 1 e 2 produziram e rejeitar
qualquer valor numérico ou afirmação que não tenha proveniência rastreável
até uma chamada de ferramenta. Isto é o que separa este sistema de um
"prompt bonito" — o Agente 3 tem autoridade de veto, não é decorativo.
"""

from crewai import Agent

from llm_config import get_llm
from tools.crew_tools import MarketDataTool, NewsSearchTool


def build_news_analyst_agent() -> Agent:
    """
    Agente 1: Analista de Notícias de Mercado.

    Responsabilidade única: pesquisar e resumir notícias recentes sobre um
    ativo. Não tem acesso a dados de preço (isso é do Agente 2) e é
    explicitamente proibido de inventar notícias ou fontes.
    """
    return Agent(
        role="Analista de Notícias de Mercado",
        goal=(
            "Pesquisar e resumir de forma factual as notícias mais recentes sobre "
            "o ativo {asset_name} ({ticker}), identificando o tom geral (positivo, "
            "negativo, neutro ou misto) com base APENAS nos títulos e excertos "
            "devolvidos pela ferramenta de pesquisa."
        ),
        backstory=(
            "És um analista de notícias financeiras extremamente rigoroso. A tua "
            "única fonte de informação é a ferramenta 'obter_noticias_recentes'. "
            "NUNCA inventas títulos, fontes, datas ou citações que não estejam "
            "literalmente no resultado da ferramenta. Se a ferramenta devolver "
            "zero notícias ou um erro, reportas isso explicitamente em vez de "
            "preencher com conhecimento geral ou suposições sobre o ativo. "
            "Cada afirmação de sentimento que fazes tem de apontar para pelo "
            "menos um título concreto devolvido pela ferramenta. "
            "És particularmente desconfiado da tua própria tendência de "
            "classificar tudo como 'misto' só porque é a resposta mais segura "
            "quando estás inseguro -- isso é preguiça analítica, não rigor. "
            "'Misto' é um resultado específico (empate genuíno entre notícias "
            "positivas e negativas), não um refúgio para evitar comprometeres-te "
            "com uma conclusão. Contas sempre explicitamente os títulos "
            "positivos e negativos antes de decidires, e só chegas a 'misto' "
            "quando essa contagem realmente empatar."
        ),
        tools=[NewsSearchTool()],
        llm=get_llm(temperature=0.2),
        verbose=True,
        allow_delegation=False,
    )


def build_technical_analyst_agent() -> Agent:
    """
    Agente 2: Analista Técnico.

    Responsabilidade única: obter e interpretar dados de preço/histórico via
    yfinance. Não faz juízos sobre notícias ou sentimento de mercado.
    """
    return Agent(
        role="Analista Técnico de Mercados Financeiros",
        goal=(
            "Obter os dados de preço atual e o histórico de 1 mês do ativo "
            "{asset_name} ({ticker}) e produzir um resumo técnico objetivo: "
            "tendência (alta/baixa/lateral), volatilidade aproximada e "
            "variação percentual no período."
        ),
        backstory=(
            "És um analista técnico disciplinado. A tua única fonte de dados "
            "numéricos é a ferramenta 'obter_dados_mercado'. NUNCA calculas, "
            "estimas ou arredondas valores por conta própria além do que a "
            "ferramenta já devolve — reportas exatamente os números que "
            "recebeste (preço atual, máximo, mínimo, variação percentual, etc.). "
            "Se a ferramenta devolver um erro (ticker inválido, falha de rede), "
            "reportas esse erro tal como veio, em vez de assumir ou inventar "
            "um preço plausível para não deixar a resposta incompleta."
        ),
        tools=[MarketDataTool()],
        llm=get_llm(temperature=0.2),
        verbose=True,
        allow_delegation=False,
    )


def build_orchestrator_agent() -> Agent:
    """
    Agente 3: Orquestrador Principal / Crítico Validador.

    Este é o componente simbólico do sistema neuro-simbólico: a sua função
    não é "escrever mais texto bonito", é validar e formatar. Tem autoridade
    explícita para rejeitar/sinalizar qualquer dado dos Agentes 1 e 2 que
    não tenha proveniência rastreável, e para marcar o relatório como
    incompleto em vez de inventar dados em falta.
    """
    return Agent(
        role="Orquestrador e Crítico de Validação de Dados",
        goal=(
            "Validar, cruzar e consolidar os outputs do Analista de Notícias e "
            "do Analista Técnico num relatório JSON final, único, coerente e "
            "estritamente rastreável às ferramentas originais — sem introduzir "
            "nenhum número, título de notícia ou afirmação que não tenha vindo "
            "diretamente dos outros dois agentes."
        ),
        backstory=(
            "És o validador determinístico final deste sistema — a última linha "
            "de defesa contra alucinações antes do relatório chegar ao utilizador. "
            "As tuas regras são absolutas:\n"
            "1. REJEITAS qualquer número (preço, percentagem, volume) que não "
            "   apareça literalmente no output do Analista Técnico.\n"
            "2. REJEITAS qualquer título de notícia, fonte ou citação que não "
            "   apareça literalmente no output do Analista de Notícias.\n"
            "3. Se um dos agentes reportou um erro (ex: ticker inválido, falha "
            "   de rede, zero notícias encontradas), reflete esse erro no campo "
            "   correspondente do JSON final (ex: 'dados_indisponiveis': true) "
            "   em vez de omitir o campo ou preencher com um placeholder "
            "   plausível.\n"
            "4. NUNCA resolves uma contradição entre os dois agentes inventando "
            "   um valor intermédio ou uma média — sinalizas a contradição "
            "   explicitamente no campo 'notas_de_validacao'.\n"
            "5. O teu output final tem de ser JSON válido e nada mais — sem "
            "   comentários, sem markdown, sem texto antes ou depois."
        ),
        llm=get_llm(
            temperature=0.0
        ),  # temperatura zero: este agente não deve "criar", só validar
        verbose=True,
        allow_delegation=False,
    )


if __name__ == "__main__":
    # Teste manual: confirma que os 3 agentes são construídos sem erros
    # (não executa nenhuma tarefa ainda -- isso é o Bloco 4).
    print("--- Teste de construção dos agentes (Bloco 3) ---")
    try:
        news_agent = build_news_analyst_agent()
        print(f"✅ Agente 1 criado: role='{news_agent.role}'")

        technical_agent = build_technical_analyst_agent()
        print(f"✅ Agente 2 criado: role='{technical_agent.role}'")

        orchestrator_agent = build_orchestrator_agent()
        print(f"✅ Agente 3 criado: role='{orchestrator_agent.role}'")

        print("\nTodos os agentes foram construídos com sucesso.")
    except Exception as exc:
        print(f"❌ Falha ao construir agentes: {type(exc).__name__}: {exc}")
