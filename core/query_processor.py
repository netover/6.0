"""
Query Processor - Processamento inteligente de queries do usuário.

Este módulo analisa queries, classifica por tipo, extrai entidades e rankeia
contexto relevante para fornecer ao LLM respostas mais precisas.

Features:
- Classificação automática de query (status, troubleshoot, how-to, etc)
- Extração de entidades (jobs, comandos, status codes)
- Ranking de contexto por relevância
- Templates de prompt otimizados por tipo de query
"""

from __future__ import annotations

import logging
import re
from enum import Enum

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS E MODELS
# =============================================================================


class QueryType(str, Enum):
    """Tipos de query identificados."""

    STATUS = "status"  # "Status do job X?"
    TROUBLESHOOT = "troubleshoot"  # "Por que job X falhou?"
    HOW_TO = "how_to"  # "Como fazer X?"
    DEFINITION = "definition"  # "O que é X?"
    COMPARISON = "comparison"  # "Diferença entre X e Y?"
    GENERAL = "general"  # Outros casos


class StructuredQuery(BaseModel):
    """Query estruturada com contexto rankeado."""

    original_query: str
    query_type: QueryType
    entities: list[str]
    intent: str
    ranked_context: list[dict]


# =============================================================================
# QUERY PROCESSOR
# =============================================================================


class QueryProcessor:
    """
    Processa queries do usuário e estrutura para o LLM.

    Fluxo:
    1. Classifica tipo de query
    2. Extrai entidades (jobs, comandos, etc)
    3. Busca contexto relevante
    4. Rankeia contexto por relevância
    5. Formata prompt estruturado
    """

    # Palavras-chave para classificação rápida
    STATUS_KEYWORDS = ["status", "estado", "situação", "está", "rodando", "executando"]
    TROUBLESHOOT_KEYWORDS = [
        "por que",
        "porque",
        "falhou",
        "erro",
        "problema",
        "abend",
        "investigar",
        "diagnosticar",
    ]
    HOW_TO_KEYWORDS = ["como", "fazer", "configurar", "criar", "executar", "rodar"]
    DEFINITION_KEYWORDS = ["o que é", "que é", "significa", "definição", "explicar"]
    COMPARISON_KEYWORDS = ["diferença", "comparar", "versus", "vs", "entre"]

    def __init__(self, llm_service, knowledge_graph):
        """
        Inicializa o processador.

        Args:
            llm_service: Serviço LLM para classificação avançada
            knowledge_graph: Knowledge Graph para contexto
        """
        self.llm = llm_service
        self.kg = knowledge_graph

    async def process_query(self, query: str) -> StructuredQuery:
        """
        Processa query e retorna estrutura.

        Args:
            query: Query do usuário

        Returns:
            Query estruturada
        """
        # 1. Classificar tipo de query
        query_type = self._classify_query_fast(query)

        # 2. Extrair entidades
        entities = self._extract_entities(query)

        # 3. Buscar contexto
        try:
            raw_context = await self.kg.get_relevant_context(query)
        except Exception as e:
            logger.warning("Failed to get KG context: %s", e)
            raw_context = ""

        # 4. Rankear contexto
        ranked_context = self._rank_context(query, raw_context, entities)

        # 5. Gerar intent
        intent = self._generate_intent(query, query_type, entities)

        return StructuredQuery(
            original_query=query,
            query_type=query_type,
            entities=entities,
            intent=intent,
            ranked_context=ranked_context,
        )

    def _classify_query_fast(self, query: str) -> QueryType:
        """
        Classifica tipo de query usando regras (rápido).

        Args:
            query: Query do usuário

        Returns:
            Tipo da query
        """
        query_lower = query.lower()

        # Verificar cada tipo em ordem de prioridade
        if any(kw in query_lower for kw in self.TROUBLESHOOT_KEYWORDS):
            return QueryType.TROUBLESHOOT

        if any(kw in query_lower for kw in self.STATUS_KEYWORDS):
            return QueryType.STATUS

        if any(kw in query_lower for kw in self.HOW_TO_KEYWORDS):
            return QueryType.HOW_TO

        if any(kw in query_lower for kw in self.DEFINITION_KEYWORDS):
            return QueryType.DEFINITION

        if any(kw in query_lower for kw in self.COMPARISON_KEYWORDS):
            return QueryType.COMPARISON

        return QueryType.GENERAL

    def _extract_entities(self, query: str) -> list[str]:
        """
        Extrai entidades da query (jobs, comandos, etc).

        Args:
            query: Query do usuário

        Returns:
            Lista de entidades extraídas
        """
        entities = []

        # Pattern 1: Nomes de jobs (maiúsculas com underscores)
        job_pattern = r"\b[A-Z][A-Z0-9_]{2,}\b"
        entities.extend(re.findall(job_pattern, query))

        # Pattern 2: Comandos TWS (começam com conman, jobman, etc)
        cmd_pattern = r"\b(conman|jobman|planman)\s+\w+"
        entities.extend(re.findall(cmd_pattern, query, re.IGNORECASE))

        # Pattern 3: Status codes (ABEND, SUCC, etc)
        status_pattern = r"\b(ABEND|SUCC|READY|WAIT|EXEC|ERROR|PEND)\b"
        entities.extend(re.findall(status_pattern, query, re.IGNORECASE))

        # Pattern 4: Workstation names
        ws_pattern = r"\b(PROD|TEST|DEV|QA)\b"
        entities.extend(re.findall(ws_pattern, query))

        return list(set(entities))  # Remove duplicatas

    def _rank_context(self, query: str, raw_context: str, entities: list[str]) -> list[dict]:
        """
        Rankeia contexto por relevância.

        Estratégia de scoring:
        1. Snippets que mencionam entidades = +10 pontos
        2. Snippets recentes (últimas 24h) = +5 pontos
        3. Snippets com solução bem-sucedida = +3 pontos
        4. Similaridade semântica com query = 0-10 pontos

        Args:
            query: Query original
            raw_context: Contexto bruto do KG
            entities: Entidades extraídas

        Returns:
            Lista de snippets rankeados
        """
        if not raw_context:
            return []

        # Quebrar contexto em snippets
        snippets = [s.strip() for s in raw_context.split("\n\n") if s.strip()]

        ranked = []

        for snippet in snippets:
            score = 0.0

            # Score 1: Menciona entidades?
            for entity in entities:
                if entity.lower() in snippet.lower():
                    score += 10

            # Score 2: Recente?
            if any(kw in snippet.lower() for kw in ["24h", "hoje", "hoje", "recente"]):
                score += 5

            # Score 3: Solução bem-sucedida?
            if any(
                kw in snippet.lower()
                for kw in ["resolvido", "sucesso", "solucionado", "corrigido"]
            ):
                score += 3

            # Score 4: Similaridade semântica (simplificado)
            query_words = set(query.lower().split())
            snippet_words = set(snippet.lower().split())
            common_words = query_words & snippet_words
            score += len(common_words) / 2

            ranked.append({"content": snippet, "score": score})

        # Ordenar por score
        ranked.sort(key=lambda x: x["score"], reverse=True)

        # Retornar top 5
        return ranked[:5]

    def _generate_intent(self, query: str, query_type: QueryType, entities: list[str]) -> str:
        """
        Gera intent resumido.

        Args:
            query: Query original
            query_type: Tipo da query
            entities: Entidades extraídas

        Returns:
            Intent resumido
        """
        entity_str = ", ".join(entities) if entities else "sistema TWS"

        intent_templates = {
            QueryType.STATUS: f"Verificar status de {entity_str}",
            QueryType.TROUBLESHOOT: f"Diagnosticar problema em {entity_str}",
            QueryType.HOW_TO: f"Aprender a realizar operação com {entity_str}",
            QueryType.DEFINITION: f"Entender conceito: {entity_str}",
            QueryType.COMPARISON: f"Comparar: {entity_str}",
            QueryType.GENERAL: f"Consulta geral sobre {entity_str}",
        }

        return intent_templates.get(query_type, f"Consulta sobre {entity_str}")

    def format_for_llm(self, structured: StructuredQuery) -> list[dict]:
        """
        Formata query estruturada para o LLM.

        Retorna messages no formato OpenAI.

        Args:
            structured: Query estruturada

        Returns:
            Lista de messages para o LLM
        """
        # System prompt adaptado ao tipo de query
        system_prompts = {
            QueryType.STATUS: """Você é um assistente especializado em monitoramento TWS/HWA.

Quando responder sobre status:
1. Seja objetivo e direto
2. Destaque o estado atual claramente
3. Mencione última execução se relevante
4. Sugira próximos passos se houver problemas

Use português brasileiro claro e profissional.""",
            QueryType.TROUBLESHOOT: """Você é um especialista em troubleshooting TWS/HWA.

Ao diagnosticar problemas:
1. Analise o contexto histórico fornecido
2. Identifique padrões de falhas similares
3. Liste causas possíveis em ordem de probabilidade
4. Forneça passos concretos de resolução
5. Mencione se a solução já funcionou antes

Use português brasileiro claro e profissional.""",
            QueryType.HOW_TO: """Você é um instrutor de operações TWS/HWA.

Ao ensinar procedimentos:
1. Liste pré-requisitos se houver
2. Forneça passos numerados e claros
3. Inclua comandos exatos quando aplicável
4. Alerte sobre possíveis erros comuns
5. Sugira como validar o resultado

Use português brasileiro claro e profissional.""",
            QueryType.DEFINITION: """Você é um glossário TWS/HWA interativo.

Ao explicar conceitos:
1. Defina de forma clara e concisa
2. Dê exemplos práticos
3. Relacione com conceitos similares quando relevante
4. Mencione casos de uso comuns

Use português brasileiro claro e profissional.""",
            QueryType.COMPARISON: """Você é um comparador técnico TWS/HWA.

Ao comparar conceitos:
1. Liste semelhanças primeiro
2. Liste diferenças claramente
3. Indique quando usar cada um
4. Dê exemplos de cada caso se possível

Use português brasileiro claro e profissional.""",
            QueryType.GENERAL: """Você é um assistente TWS/HWA experiente.

Responda de forma clara, objetiva e profissional.
Use português brasileiro.""",
        }

        system_prompt = system_prompts.get(
            structured.query_type, "Você é um assistente TWS/HWA. Responda de forma clara."
        )

        # Contexto rankeado (top 3)
        if structured.ranked_context:
            context_str = "\n\n".join(
                [f"[Relevância: {c['score']:.1f}]\n{c['content']}" for c in structured.ranked_context[:3]]
            )
        else:
            context_str = ""

        # User message estruturado
        if context_str:
            user_message = f"""**Contexto de soluções e informações anteriores (rankeado por relevância):**
{context_str}

**Pergunta do usuário:**
{structured.original_query}

Por favor, responda considerando o contexto acima quando relevante."""
        else:
            user_message = structured.original_query

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "QueryType",
    "StructuredQuery",
    "QueryProcessor",
]
