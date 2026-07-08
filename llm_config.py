"""
Configuração do Motor LLM — usado por todos os agentes (Bloco 3+).

Usamos `crewai.LLM`, o wrapper nativo do CrewAI (assente em `litellm`), em
vez de instanciar `langchain-groq`/`langchain-ollama` diretamente. Vantagens:
  1. Menos dependências (uma classe serve Groq, Ollama, OpenAI, etc.)
  2. Integração direta com `Agent(llm=...)` no Bloco 3, sem camadas extra
  3. Trocar de fornecedor (Groq -> Ollama local) é só mudar a string do
     modelo (ex: "groq/openai/gpt-oss-120b" -> "ollama/llama3"), sem
     reescrever nada nos agentes.
"""

import logging
import os

from crewai import LLM
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tools.exceptions import FinancialToolError

logger = logging.getLogger(__name__)

load_dotenv()  # lê o ficheiro .env (se existir) para variáveis como GROQ_API_KEY

# --- Workaround: bug conhecido do crewAI (#5886), ainda aberto em julho 2026 ---
# O crewAI injeta um campo "cache_breakpoint" em todas as mensagens enviadas ao
# LLM, mas só o adapter da Anthropic sabe tratar/remover esse campo antes de o
# enviar à API. Providers OpenAI-compatible (incluindo Groq) rejeitam o campo
# com um erro 400 ("property 'cache_breakpoint' is unsupported"). O próprio
# issue do crewAI sugere este monkey-patch como workaround enquanto não sai
# uma correção oficial: https://github.com/crewAIInc/crewAI/issues/5886
# TODO: remover este patch quando o crewAI lançar uma versão com a correção.
try:
    import crewai.llms.cache as _crewai_cache

    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
    logger.debug("Workaround para crewAI #5886 (cache_breakpoint) aplicado.")
except ImportError:
    logger.warning(
        "Não foi possível aplicar o workaround para crewAI #5886 "
        "(módulo crewai.llms.cache não encontrado -- pode já estar corrigido "
        "nesta versão, ou a estrutura interna do crewAI mudou)."
    )


class LLMConfigError(FinancialToolError):
    """Lançado quando o LLM não está configurado corretamente ou não responde."""


# Erros de rede/rate-limit que justificam retry automático (não erros de config/auth).
_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError)


def get_llm(temperature: float = 0.2) -> LLM:
    """
    Constrói e devolve a instância de LLM configurada para Groq.

    Usamos uma temperatura baixa (0.2) por defeito porque este é um sistema
    de análise financeira — queremos respostas consistentes e factuais,
    não criativas. Isto é especialmente importante para o Agente 3
    (Orquestrador/Crítico), que valida dados e não deve "improvisar".

    Args:
        temperature: controla a aleatoriedade das respostas (0.0 = determinístico).

    Returns:
        Instância de `crewai.LLM` pronta a passar a um `Agent(llm=...)`.

    Raises:
        LLMConfigError: se a variável de ambiente GROQ_API_KEY não estiver definida.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "cola_aqui_a_tua_chave_groq":
        raise LLMConfigError(
            "GROQ_API_KEY não está definida ou ainda tem o valor placeholder. "
            "Copia '.env.example' para '.env' e preenche com a tua chave real de "
            "https://console.groq.com/keys"
        )

    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    logger.info("A configurar LLM: provider=groq, model=%s, temperature=%s", model_name, temperature)

    try:
        return LLM(
            model=f"groq/{model_name}",
            api_key=api_key,
            temperature=temperature,
        )
    except ImportError as exc:
        raise LLMConfigError(
            f"Falta uma dependência para inicializar o LLM: {exc}. "
            "Corre 'pip install -r requirements.txt' novamente (litellm em falta?)."
        ) from exc


@retry(
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=15),
    reraise=True,
)
def _call_llm(llm: LLM, prompt: str) -> str:
    """Wrapper de baixo nível com retry para falhas de rede transitórias."""
    return llm.call(prompt)


def test_llm_connection() -> str:
    """
    Envia um prompt de teste simples ao LLM configurado e valida a resposta.

    Este teste existe para sermos apanhados aqui, no Bloco 2, e não só
    quando já tivermos 3 agentes e tarefas montadas no Bloco 4 — mais fácil
    de depurar um problema de conectividade/chave isoladamente.

    Returns:
        O texto da resposta do LLM.

    Raises:
        LLMConfigError: se a chave não estiver configurada, a ligação falhar
                        após retries, ou a resposta vier vazia/inesperada.
    """
    llm = get_llm(temperature=0.0)
    prompt = (
        "Responde apenas com a palavra OK, sem mais nenhum texto, "
        "para confirmar que a ligação está a funcionar."
    )

    try:
        response = _call_llm(llm, prompt)
    except _RETRYABLE_EXCEPTIONS as exc:
        raise LLMConfigError(
            f"Falha de rede ao contactar a API da Groq após múltiplas tentativas: {exc}"
        ) from exc
    except Exception as exc:
        # litellm/Groq costumam lançar exceções específicas (ex: AuthenticationError,
        # RateLimitError) que não estão tipadas aqui de propósito -- preferimos
        # apanhar tudo e reportar com contexto, para o utilizador perceber já
        # se é a chave, o modelo, ou a rede.
        raise LLMConfigError(f"Erro ao chamar o LLM (Groq): {type(exc).__name__}: {exc}") from exc

    if not response or not response.strip():
        raise LLMConfigError(
            "O LLM respondeu, mas a resposta veio vazia. Verifica o modelo configurado "
            "em GROQ_MODEL e confirma que ainda está ativo em https://console.groq.com/docs/models"
        )

    return response.strip()


if __name__ == "__main__":
    print("--- Teste de ligação ao LLM (Groq) ---")
    try:
        resposta = test_llm_connection()
        print(f"✅ LLM respondeu corretamente: '{resposta}'")
    except LLMConfigError as e:
        print(f"❌ [LLMConfigError] {e}")
