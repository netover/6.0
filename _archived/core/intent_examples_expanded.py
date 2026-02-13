"""
Exemplos Expandidos de Intent para Embedding Router.

Este módulo contém 200+ exemplos de queries organizados por intent,
usados para classificação baseada em embeddings.

Versão: 5.3.18
Target: 200+ exemplos (14+ por intent)

Idiomas: Português (BR) e Inglês
"""

from resync.core.embedding_router import RouterIntent

# =============================================================================
# EXEMPLOS EXPANDIDOS (200+)
# =============================================================================

INTENT_EXAMPLES_EXPANDED = {
    RouterIntent.DEPENDENCY_CHAIN: [
        # Português
        "quais são as dependências do job X",
        "mostre a cadeia de dependências",
        "lista predecessores do job",
        "quais jobs rodam antes de X",
        "dependências upstream do job",
        "jobs que precisam terminar antes",
        "qual a árvore de dependências",
        "predecessores diretos e indiretos",
        "cadeia completa de jobs",
        "fluxo de execução antes do job",
        "jobs anteriores na cadeia",
        "de que jobs X depende",
        "requisitos para executar o job",
        "ordem de execução dos jobs",
        "sequência de dependências",
        # English
        "show job dependencies",
        "what runs before this job",
        "predecessor jobs list",
        "upstream dependencies",
        "job execution chain",
        "dependency tree",
        "jobs required before X",
    ],
    RouterIntent.IMPACT_ANALYSIS: [
        # Português
        "qual o impacto se o job falhar",
        "quantos jobs serão afetados",
        "análise de impacto do job",
        "jobs dependentes downstream",
        "cascata de falha do job",
        "efeito dominó se parar",
        "jobs que vão atrasar",
        "impacto no schedule",
        "consequências da falha",
        "análise de risco do job",
        "o que acontece se X falhar",
        "jobs afetados pela falha",
        "propagação do erro",
        "impacto em outros processos",
        "quais schedules serão afetados",
        # English
        "impact if job fails",
        "downstream affected jobs",
        "failure cascade analysis",
        "risk assessment for job",
        "what breaks if this fails",
        "jobs impacted by failure",
        "domino effect analysis",
    ],
    RouterIntent.RESOURCE_CONFLICT: [
        # Português
        "conflito de recursos",
        "jobs usando mesmo recurso",
        "contenção de recursos",
        "recursos exclusivos em uso",
        "deadlock de recursos",
        "recursos compartilhados",
        "jobs competindo por recurso",
        "alocação de recursos",
        "recurso bloqueado por job",
        "liberação de recursos",
        "qual job está usando o recurso",
        "recursos ocupados",
        "disputa por recursos",
        "recursos disponíveis",
        "fila de recursos",
        # English
        "resource conflict detection",
        "shared resource contention",
        "exclusive resource lock",
        "resource allocation issues",
        "jobs competing for resource",
        "resource deadlock",
        "resource availability",
    ],
    RouterIntent.CRITICAL_JOBS: [
        # Português
        "jobs críticos do dia",
        "jobs prioritários",
        "jobs que não podem falhar",
        "SLA críticos",
        "jobs de alta prioridade",
        "processos essenciais",
        "jobs mandatórios",
        "batch crítico",
        "jobs com deadline",
        "processos regulatórios",
        "jobs mais importantes",
        "prioridade máxima",
        "jobs de missão crítica",
        "processos críticos hoje",
        "jobs sensíveis a atraso",
        # English
        "critical jobs today",
        "high priority jobs",
        "SLA critical processes",
        "mandatory batch jobs",
        "deadline sensitive jobs",
        "mission critical jobs",
        "top priority processes",
    ],
    RouterIntent.JOB_LINEAGE: [
        # Português
        "linhagem do job",
        "histórico de execuções",
        "evolução do job",
        "versões anteriores",
        "mudanças no job",
        "quem criou o job",
        "audit trail do job",
        "modificações recentes",
        "origem do job",
        "rastreabilidade",
        "quando foi criado",
        "alterações no job",
        "histórico de mudanças",
        "quem alterou o job",
        "log de alterações",
        # English
        "job lineage",
        "execution history",
        "job audit trail",
        "who created this job",
        "job change history",
        "modification log",
        "job versioning",
    ],
    RouterIntent.TROUBLESHOOTING: [
        # Português
        "como resolver erro X",
        "job falhou, o que fazer",
        "debug do job",
        "investigar falha",
        "análise de erro",
        "por que o job falhou",
        "solução para RC 12",
        "corrigir abend",
        "recuperar job",
        "restart após falha",
        "job preso, como resolver",
        "timeout do job",
        "job lento, como otimizar",
        "erro de conexão TWS",
        "problema de permissão",
        "como reiniciar job",
        "job não inicia",
        "job travou",
        "cancelar job",
        "forçar término",
        # English
        "how to fix job error",
        "job failed what to do",
        "debug job failure",
        "troubleshoot RC code",
        "fix abend job",
        "recover failed job",
        "restart job",
    ],
    RouterIntent.ERROR_LOOKUP: [
        # Português
        "o que significa RC 8",
        "código de erro 12",
        "traduzir erro TWS",
        "significado do abend",
        "erro AWKR0001",
        "código de retorno",
        "mensagem de erro",
        "catálogo de erros",
        "lista de RCs",
        "erro desconhecido",
        "explicar código de erro",
        "tabela de erros TWS",
        "RC 4 significa o que",
        "interpretar mensagem",
        "dicionário de erros",
        # English
        "what does RC 8 mean",
        "error code lookup",
        "TWS error message",
        "return code meaning",
        "abend code translation",
        "error catalog",
        "decode error message",
    ],
    RouterIntent.DOCUMENTATION: [
        # Português
        "documentação do TWS",
        "manual do job",
        "como usar ferramenta X",
        "guia de referência",
        "tutorial TWS",
        "procedimento padrão",
        "boas práticas",
        "instruções de operação",
        "onde encontro informação",
        "referência técnica",
        "documentação do comando",
        "help do TWS",
        "guia do usuário",
        "manual de operações",
        "especificação técnica",
        # English
        "TWS documentation",
        "job manual",
        "how to use TWS",
        "reference guide",
        "best practices",
        "operation instructions",
        "user guide",
    ],
    RouterIntent.EXPLANATION: [
        # Português
        "explique como funciona",
        "o que é isso",
        "como funciona o TWS",
        "para que serve",
        "conceito de batch",
        "definição de schedule",
        "explique dependências",
        "o que significa workstation",
        "diferença entre X e Y",
        "conceitos básicos",
        "introdução ao TWS",
        "fundamentos de scheduling",
        "como funciona o agendamento",
        "explicar recursos exclusivos",
        "o que é um job stream",
        # English
        "explain how it works",
        "what is this",
        "TWS concepts",
        "define batch job",
        "explain scheduling",
        "what does X mean",
        "fundamentals",
    ],
    RouterIntent.JOB_DETAILS: [
        # Português
        "status do job X",
        "detalhes do job",
        "informações do job",
        "quando rodou o job",
        "último run do job",
        "próxima execução",
        "parâmetros do job",
        "configuração do job",
        "owner do job",
        "workstation do job",
        "horário agendado",
        "duração média",
        "estatísticas do job",
        "RC do último run",
        "log do job",
        "qual o status atual",
        "job está rodando",
        "job finalizou",
        "tempo de execução",
        "histórico de runs",
        # English
        "job status",
        "job details",
        "when did job run",
        "next scheduled run",
        "job parameters",
        "job configuration",
        "execution time",
    ],
    RouterIntent.ROOT_CAUSE: [
        # Português
        "causa raiz do problema",
        "por que falhou",
        "análise de causa",
        "investigação profunda",
        "origem do erro",
        "motivo da falha",
        "diagnóstico completo",
        "análise forense",
        "o que causou o abend",
        "fonte do problema",
        "investigar causa raiz",
        "determinar origem",
        "análise detalhada",
        "entender o problema",
        "raiz da falha",
        # English
        "root cause analysis",
        "why did it fail",
        "failure investigation",
        "problem diagnosis",
        "error origin",
        "deep analysis",
        "determine cause",
    ],
    RouterIntent.GENERAL: [
        # Português
        "ajuda geral",
        "o que você pode fazer",
        "quais funcionalidades",
        "como posso usar",
        "preciso de ajuda",
        "não sei por onde começar",
        "opções disponíveis",
        "o que você sabe",
        "capacidades do sistema",
        "menu de opções",
        # English
        "general help",
        "what can you do",
        "available features",
        "help me",
        "getting started",
    ],
    RouterIntent.GREETING: [
        # Português
        "olá",
        "oi",
        "bom dia",
        "boa tarde",
        "boa noite",
        "e aí",
        "tudo bem",
        "como vai",
        "opa",
        "eae",
        # English
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
    ],
    RouterIntent.CHITCHAT: [
        # Português
        "como você está",
        "tudo certo",
        "obrigado",
        "valeu",
        "até mais",
        "tchau",
        "legal",
        "entendi",
        "ok",
        "beleza",
        # English
        "how are you",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
    ],
}


def get_expanded_examples():
    """
    Retorna os exemplos expandidos.

    Returns:
        Dict mapeando RouterIntent para lista de exemplos
    """
    return INTENT_EXAMPLES_EXPANDED


def get_example_stats():
    """
    Retorna estatísticas dos exemplos.

    Returns:
        Dict com estatísticas
    """
    stats = {
        "total_intents": len(INTENT_EXAMPLES_EXPANDED),
        "total_examples": 0,
        "by_intent": {},
        "min_examples": float("inf"),
        "max_examples": 0,
    }

    for intent, examples in INTENT_EXAMPLES_EXPANDED.items():
        count = len(examples)
        stats["total_examples"] += count
        stats["by_intent"][intent.value] = count
        stats["min_examples"] = min(stats["min_examples"], count)
        stats["max_examples"] = max(stats["max_examples"], count)

    stats["avg_examples"] = stats["total_examples"] / stats["total_intents"]

    return stats


def validate_examples():
    """
    Valida os exemplos de intent.

    Raises:
        AssertionError se validação falhar
    """
    stats = get_example_stats()

    # Validações
    assert stats["total_examples"] >= 200, (
        f"Precisa de 200+ exemplos, tem {stats['total_examples']}"
    )

    assert stats["min_examples"] >= 10, (
        f"Cada intent precisa de 10+ exemplos, mínimo atual: {stats['min_examples']}"
    )

    # Verificar duplicados
    all_examples = []
    for examples in INTENT_EXAMPLES_EXPANDED.values():
        all_examples.extend([e.lower() for e in examples])

    duplicates = set([x for x in all_examples if all_examples.count(x) > 1])
    assert len(duplicates) == 0, f"Duplicados encontrados: {duplicates}"

    print("✅ Validação passou!")
    print(f"   Total de intents: {stats['total_intents']}")
    print(f"   Total de exemplos: {stats['total_examples']}")
    print("   Média por intent: {stats['avg_examples']:.1f}")
    print(f"   Min/Max: {stats['min_examples']}/{stats['max_examples']}")

    return True


def merge_with_existing():
    """
    Mescla exemplos expandidos com os existentes no router.

    Útil para atualizar o router sem perder exemplos existentes.

    Returns:
        Dict com exemplos mesclados
    """
    from resync.core.embedding_router import INTENT_EXAMPLES

    merged = {}

    for intent in RouterIntent:
        existing = set(INTENT_EXAMPLES.get(intent, []))
        expanded = set(INTENT_EXAMPLES_EXPANDED.get(intent, []))
        merged[intent] = list(existing | expanded)

    return merged


if __name__ == "__main__":
    # Executar validação
    validate_examples()

    # Mostrar estatísticas
    stats = get_example_stats()
    print("\n📊 Estatísticas por Intent:")
    for intent, count in sorted(stats["by_intent"].items(), key=lambda x: -x[1]):
        bar = "█" * (count // 2)
        print("   {intent:20} {count:3} {bar}")
