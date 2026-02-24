"""
TWS RAG Integration - Queries Históricas em Linguagem Natural

Este módulo permite fazer perguntas sobre o histórico do TWS
usando linguagem natural, como:
- "O que aconteceu ontem?"
- "Quais jobs falharam na semana passada?"
- "Tem algum padrão de falha no job X?"

Integra com o Context Store e Status Store para RAG.

Autor: Resync Team
Versão: 5.2
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class QueryIntent:
    """Intent extraído de uma query."""

    intent_type: (
        str  # summary, failures, patterns, job_history, workstation, comparison
    )
    time_range: tuple[datetime, datetime]
    entities: list[str]  # job names, workstation names
    filters: dict[str, Any]
    original_query: str


@dataclass
class QueryResult:
    """Resultado de uma query RAG."""

    success: bool
    summary: str
    details: list[dict[str, Any]]
    suggestions: list[str]
    metadata: dict[str, Any]


class TWSQueryProcessor:
    """
    Processa queries em linguagem natural sobre o TWS.

    Extrai intent, busca dados relevantes e gera resposta.
    """

    # Padrões de tempo
    TIME_PATTERNS = {
        r"ontem": lambda _m=None: (
            datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
            - timedelta(days=1),
            datetime.now(timezone.utc).replace(hour=0, minute=0, second=0),
        ),
        r"hoje": lambda _m=None: (
            datetime.now(timezone.utc).replace(hour=0, minute=0, second=0),
            datetime.now(timezone.utc),
        ),
        r"esta semana|essa semana|semana atual": lambda _m=None: (
            datetime.now(timezone.utc)
            - timedelta(days=datetime.now(timezone.utc).weekday()),
            datetime.now(timezone.utc),
        ),
        r"semana passada|última semana": lambda _m=None: (
            datetime.now(timezone.utc)
            - timedelta(days=datetime.now(timezone.utc).weekday() + 7),
            datetime.now(timezone.utc)
            - timedelta(days=datetime.now(timezone.utc).weekday()),
        ),
        r"este mês|esse mês|mês atual": lambda _m=None: (
            datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0),
            datetime.now(timezone.utc),
        ),
        r"últimos? (\d+) dias?": lambda m: (
            datetime.now(timezone.utc) - timedelta(days=int(m.group(1))),
            datetime.now(timezone.utc),
        ),
        r"últimas? (\d+) horas?": lambda m: (
            datetime.now(timezone.utc) - timedelta(hours=int(m.group(1))),
            datetime.now(timezone.utc),
        ),
        r"últimos? (\d+) minutos?": lambda m: (
            datetime.now(timezone.utc) - timedelta(minutes=int(m.group(1))),
            datetime.now(timezone.utc),
        ),
    }

    # Padrões de intent
    INTENT_PATTERNS = {
        "failures": [
            r"falh(ou|aram|as?)",
            r"abend",
            r"erro(s)?",
            r"problema(s)?",
            r"deu ruim",
        ],
        "summary": [
            r"o que aconteceu",
            r"resumo",
            r"sumário",
            r"overview",
            r"visão geral",
            r"como est(á|ava)",
        ],
        "patterns": [
            r"padr(ão|ões)",
            r"tend(ência|ências)",
            r"recorrente",
            r"frequente",
            r"sempre",
            r"costuma",
        ],
        "job_history": [
            r"histórico (do|da|de) job",
            r"job .+ (rodou|executou|falhou)",
            r"quando .+ (rodou|executou)",
        ],
        "workstation": [
            r"workstation",
            r"ws ",
            r"servidor",
            r"agente",
        ],
        "comparison": [
            r"compar(ar|ação)",
            r"diferença",
            r"melhorou|piorou",
            r"versus|vs",
        ],
    }

    def __init__(self, status_store: Any = None):
        """
        Inicializa o processador.

        Args:
            status_store: TWSStatusStore para queries
        """
        self.status_store = status_store

    async def process_query(self, query: str) -> QueryResult:
        """
        Processa uma query em linguagem natural.

        Args:
            query: Pergunta em linguagem natural

        Returns:
            Resultado da query
        """
        # 1. Extrai intent
        intent = self._extract_intent(query)

        logger.info(
            "query_processed",
            intent_type=intent.intent_type,
            time_range=(
                intent.time_range[0].isoformat(),
                intent.time_range[1].isoformat(),
            ),
            entities=intent.entities,
        )

        # 2. Busca dados baseado no intent
        if intent.intent_type == "summary":
            return await self._handle_summary_query(intent)
        if intent.intent_type == "failures":
            return await self._handle_failures_query(intent)
        if intent.intent_type == "patterns":
            return await self._handle_patterns_query(intent)
        if intent.intent_type == "job_history":
            return await self._handle_job_history_query(intent)
        if intent.intent_type == "workstation":
            return await self._handle_workstation_query(intent)
        if intent.intent_type == "comparison":
            return await self._handle_comparison_query(intent)
        return await self._handle_general_query(intent)

    def _extract_intent(self, query: str) -> QueryIntent:
        """Extrai intent da query."""
        query_lower = query.lower()

        # Extrai range de tempo
        time_range = self._extract_time_range(query_lower)

        # Extrai tipo de intent
        intent_type = "general"
        for itype, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    intent_type = itype
                    break
            if intent_type != "general":
                break

        # Extrai entidades (nomes de jobs, workstations)
        entities = self._extract_entities(query)

        return QueryIntent(
            intent_type=intent_type,
            time_range=time_range,
            entities=entities,
            filters={},
            original_query=query,
        )

    def _extract_time_range(self, query: str) -> tuple[datetime, datetime]:
        """Extrai range de tempo da query."""
        for pattern, time_func in self.TIME_PATTERNS.items():
            match = re.search(pattern, query)
            if match and callable(time_func):
                # Verifica se a função precisa do match
                return time_func(match)

        # Default: últimas 24 horas
        return (
            datetime.now(timezone.utc) - timedelta(hours=24),
            datetime.now(timezone.utc),
        )

    def _extract_entities(self, query: str) -> list[str]:
        """Extrai entidades (nomes de jobs, workstations)."""
        entities = []

        # Padrão para nomes de jobs (geralmente uppercase com underscores)
        job_pattern = r"\b([A-Z][A-Z0-9_]{2,})\b"
        entities.extend(re.findall(job_pattern, query))

        # Padrão para workstations
        ws_pattern = r"\b(WS[A-Z0-9]+|[A-Z]+SRV[0-9]+)\b"
        entities.extend(re.findall(ws_pattern, query, re.IGNORECASE))

        return list(set(entities))

    async def _handle_summary_query(self, intent: QueryIntent) -> QueryResult:
        """Processa query de resumo."""
        start, end = intent.time_range

        # Busca dados
        if self.status_store:
            events = await self.status_store.get_events_in_range(
                start_time=start,
                end_time=end,
                limit=500,
            )

            # Agrupa por tipo
            event_counts: dict[str, int] = {}
            for event in events:
                etype = event.get("event_type", "unknown")
                event_counts[etype] = event_counts.get(etype, 0) + 1

            # Conta falhas
            failures = [e for e in events if e.get("severity") in ["error", "critical"]]

            # Gera resumo
            period = self._format_period(start, end)

            summary = f"**Resumo {period}:**\n\n"
            summary += f"- Total de eventos: {len(events)}\n"
            summary += f"- Eventos críticos/erros: {len(failures)}\n"

            if event_counts:
                summary += "\n**Por tipo:**\n"
                for etype, count in sorted(event_counts.items(), key=lambda x: -x[1])[
                    :5
                ]:
                    summary += f"- {etype}: {count}\n"

            if failures:
                summary += "\n**Principais problemas:**\n"
                for f in failures[:5]:
                    summary += f"- {f.get('message', 'N/A')}\n"

            return QueryResult(
                success=True,
                summary=summary,
                details=events[:20],
                suggestions=[
                    "Ver detalhes de falhas",
                    "Detectar padrões",
                    "Comparar com período anterior",
                ],
                metadata={
                    "period": {"start": start.isoformat(), "end": end.isoformat()},
                    "total_events": len(events),
                    "failures": len(failures),
                },
            )

        return QueryResult(
            success=False,
            summary="Status store não disponível",
            details=[],
            suggestions=["Verificar configuração do sistema"],
            metadata={},
        )

    async def _handle_failures_query(self, intent: QueryIntent) -> QueryResult:
        """Processa query de falhas."""
        start, end = intent.time_range

        if self.status_store:
            events = await self.status_store.get_events_in_range(
                start_time=start,
                end_time=end,
                event_types=["job_abend"],
                limit=100,
            )

            # Filtra por entidades se especificadas
            if intent.entities:
                events = [
                    e
                    for e in events
                    if any(
                        ent.lower() in e.get("source", "").lower()
                        for ent in intent.entities
                    )
                ]

            # Agrupa por job
            jobs_failed: dict[str, list[dict[str, Any]]] = {}
            for event in events:
                job = event.get("source", "unknown")
                if job not in jobs_failed:
                    jobs_failed[job] = []
                jobs_failed[job].append(event)

            period = self._format_period(start, end)

            if not events:
                summary = f"🎉 Nenhuma falha encontrada {period}!"
            else:
                summary = f"**Falhas {period}:**\n\n"
                summary += f"- Total de falhas: {len(events)}\n"
                summary += f"- Jobs afetados: {len(jobs_failed)}\n\n"

                summary += "**Jobs que falharam:**\n"
                for job, failures in sorted(
                    jobs_failed.items(), key=lambda x: -len(x[1])
                )[:10]:
                    summary += f"- **{job}**: {len(failures)}x\n"
                    if failures:
                        last_error = (
                            failures[0]
                            .get("details", {})
                            .get("job", {})
                            .get("error_message")
                        )
                        if last_error:
                            summary += f"  └ Último erro: {last_error[:100]}\n"

            # Busca soluções sugeridas
            suggestions = []
            if jobs_failed and self.status_store:
                for job in list(jobs_failed.keys())[:3]:
                    solution = await self.status_store.find_solution(
                        "job_abend",
                        events[0]
                        .get("details", {})
                        .get("job", {})
                        .get("error_message", ""),
                    )
                    if solution:
                        suggestions.append(
                            f"Para {job}: {solution.get('solution', 'N/A')}"
                        )

            return QueryResult(
                success=True,
                summary=summary,
                details=events[:20],
                suggestions=suggestions
                or [
                    "Ver histórico do job",
                    "Detectar padrões de falha",
                ],
                metadata={
                    "period": {"start": start.isoformat(), "end": end.isoformat()},
                    "total_failures": len(events),
                    "jobs_affected": len(jobs_failed),
                },
            )

        return QueryResult(
            success=False,
            summary="Status store não disponível",
            details=[],
            suggestions=[],
            metadata={},
        )

    async def _handle_patterns_query(self, intent: QueryIntent) -> QueryResult:
        """Processa query de padrões."""
        if self.status_store:
            patterns = await self.status_store.get_patterns(min_confidence=0.5)

            # Filtra por entidades se especificadas
            if intent.entities:
                patterns = [
                    p
                    for p in patterns
                    if any(
                        ent.lower() in str(p.get("affected_jobs", [])).lower()
                        for ent in intent.entities
                    )
                ]

            if not patterns:
                summary = "Nenhum padrão significativo detectado no momento."
                summary += (
                    "\n\nIsso pode significar que o sistema está operando normalmente "
                )
                summary += "ou que não há dados suficientes para detectar padrões."
            else:
                summary = f"**{len(patterns)} padrões detectados:**\n\n"

                for i, p in enumerate(patterns[:5], 1):
                    summary += f"**{i}. {p.get('pattern_type', 'unknown')}** "
                    summary += f"({p.get('confidence', 0) * 100:.0f}% confiança)\n"
                    summary += f"   {p.get('description', 'N/A')}\n"
                    if p.get("suggested_action"):
                        summary += f"   💡 {p.get('suggested_action')}\n"
                    summary += "\n"

            return QueryResult(
                success=True,
                summary=summary,
                details=patterns,
                suggestions=[
                    "Investigar padrões de alta confiança",
                    "Ver histórico de jobs afetados",
                ],
                metadata={
                    "patterns_count": len(patterns),
                },
            )

        return QueryResult(
            success=False,
            summary="Status store não disponível",
            details=[],
            suggestions=[],
            metadata={},
        )

    async def _handle_job_history_query(self, intent: QueryIntent) -> QueryResult:
        """Processa query de histórico de job."""
        if not intent.entities:
            return QueryResult(
                success=False,
                summary="Por favor, especifique o nome do job para ver o histórico.",
                details=[],
                suggestions=["Exemplo: 'histórico do job BATCH_PROCESS_001'"],
                metadata={},
            )

        job_name = intent.entities[0]
        days = 7  # Default

        if self.status_store:
            history = await self.status_store.get_job_history(
                job_name=job_name,
                days=days,
                limit=50,
            )

            if not history:
                summary = (
                    "Nenhum histórico encontrado para o job "
                    f"**{job_name}** nos últimos {days} dias."
                )
            else:
                # Calcula estatísticas
                total = len(history)
                success = sum(1 for h in history if h.get("status") == "SUCC")
                failed = sum(1 for h in history if h.get("status") == "ABEND")
                success_rate = (success / total * 100) if total > 0 else 0

                # Duração média
                durations = [
                    h.get("duration_seconds")
                    for h in history
                    if h.get("duration_seconds")
                ]
                avg_duration = sum(durations) / len(durations) if durations else 0

                summary = f"**Histórico do job {job_name}** (últimos {days} dias):\n\n"
                summary += f"- Execuções: {total}\n"
                summary += f"- Sucessos: {success} ({success_rate:.1f}%)\n"
                summary += f"- Falhas: {failed}\n"
                summary += f"- Duração média: {avg_duration / 60:.1f} min\n"

                if failed > 0:
                    summary += "\n**Últimas falhas:**\n"
                    failures = [h for h in history if h.get("status") == "ABEND"][:3]
                    for f in failures:
                        error_preview = f.get("error_message", "N/A")[:50]
                        summary += (
                            f"- {f.get('timestamp', 'N/A')}: "
                            f"{error_preview}\n"
                        )

            return QueryResult(
                success=True,
                summary=summary,
                details=history[:10],
                suggestions=[
                    "Ver padrões de falha deste job",
                    "Comparar com outros jobs da mesma cadeia",
                ],
                metadata={
                    "job_name": job_name,
                    "total_executions": len(history),
                },
            )

        return QueryResult(
            success=False,
            summary="Status store não disponível",
            details=[],
            suggestions=[],
            metadata={},
        )

    async def _handle_workstation_query(self, intent: QueryIntent) -> QueryResult:
        """Processa query de workstation."""
        start, end = intent.time_range

        if self.status_store:
            events = await self.status_store.get_events_in_range(
                start_time=start,
                end_time=end,
                event_types=[
                    "workstation_offline",
                    "workstation_unlinked",
                    "ws_offline",
                    "ws_unlinked",
                ],
                limit=100,
            )

            # Filtra por entidade se especificada
            if intent.entities:
                events = [
                    e
                    for e in events
                    if any(
                        ent.lower() in e.get("source", "").lower()
                        for ent in intent.entities
                    )
                ]

            period = self._format_period(start, end)

            if not events:
                summary = f"✅ Nenhum problema com workstations {period}."
            else:
                # Agrupa por workstation
                ws_issues: dict[str, list[dict[str, Any]]] = {}
                for event in events:
                    ws = event.get("source", "unknown")
                    if ws not in ws_issues:
                        ws_issues[ws] = []
                    ws_issues[ws].append(event)

                summary = f"**Problemas com workstations {period}:**\n\n"
                summary += f"- Total de incidentes: {len(events)}\n"
                summary += f"- Workstations afetadas: {len(ws_issues)}\n\n"

                for ws, issues in sorted(ws_issues.items(), key=lambda x: -len(x[1])):
                    summary += f"- **{ws}**: {len(issues)} incidente(s)\n"

            return QueryResult(
                success=True,
                summary=summary,
                details=events[:10],
                suggestions=[
                    "Ver status atual das workstations",
                    "Verificar padrões de desconexão",
                ],
                metadata={
                    "period": {"start": start.isoformat(), "end": end.isoformat()},
                    "total_issues": len(events),
                },
            )

        return QueryResult(
            success=False,
            summary="Status store não disponível",
            details=[],
            suggestions=[],
            metadata={},
        )

    async def _handle_comparison_query(self, intent: QueryIntent) -> QueryResult:
        """Processa query de comparação."""
        start, end = intent.time_range
        period_length = end - start

        # Período anterior de mesmo tamanho
        prev_start = start - period_length
        prev_end = start

        if self.status_store:
            # Dados do período atual
            current_events = await self.status_store.get_events_in_range(
                start_time=start,
                end_time=end,
                limit=500,
            )
            current_failures = [
                e for e in current_events if e.get("severity") in ["error", "critical"]
            ]

            # Dados do período anterior
            prev_events = await self.status_store.get_events_in_range(
                start_time=prev_start,
                end_time=prev_end,
                limit=500,
            )
            prev_failures = [
                e for e in prev_events if e.get("severity") in ["error", "critical"]
            ]

            # Calcula diferenças
            event_diff = len(current_events) - len(prev_events)
            failure_diff = len(current_failures) - len(prev_failures)

            event_trend = (
                "📈 aumentou"
                if event_diff > 0
                else "📉 diminuiu"
                if event_diff < 0
                else "→ estável"
            )
            failure_trend = (
                "📈 aumentou"
                if failure_diff > 0
                else "📉 diminuiu"
                if failure_diff < 0
                else "→ estável"
            )

            summary = "**Comparação com período anterior:**\n\n"
            summary += f"**Período atual:** {self._format_period(start, end)}\n"
            summary += f"- Eventos: {len(current_events)}\n"
            summary += f"- Falhas: {len(current_failures)}\n\n"
            summary += (
                f"**Período anterior:** {self._format_period(prev_start, prev_end)}\n"
            )
            summary += f"- Eventos: {len(prev_events)}\n"
            summary += f"- Falhas: {len(prev_failures)}\n\n"
            summary += "**Tendência:**\n"
            summary += f"- Eventos: {event_trend} ({abs(event_diff):+d})\n"
            summary += f"- Falhas: {failure_trend} ({abs(failure_diff):+d})\n"

            return QueryResult(
                success=True,
                summary=summary,
                details=[],
                suggestions=[
                    "Ver detalhes das falhas",
                    "Identificar novos problemas",
                ],
                metadata={
                    "current_period": {
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                    },
                    "previous_period": {
                        "start": prev_start.isoformat(),
                        "end": prev_end.isoformat(),
                    },
                    "event_change": event_diff,
                    "failure_change": failure_diff,
                },
            )

        return QueryResult(
            success=False,
            summary="Status store não disponível",
            details=[],
            suggestions=[],
            metadata={},
        )

    async def _handle_general_query(self, intent: QueryIntent) -> QueryResult:
        """Processa query genérica."""
        # Tenta buscar eventos relevantes
        start, end = intent.time_range

        if self.status_store:
            # Busca textual se possível
            events = await self.status_store.search_events(
                intent.original_query,
                limit=20,
            )

            if events:
                summary = f'**Resultados para:** "{intent.original_query}"\n\n'
                summary += f"Encontrei {len(events)} eventos relacionados:\n\n"

                for e in events[:5]:
                    summary += (
                        f"- [{e.get('timestamp', 'N/A')}] {e.get('message', 'N/A')}\n"
                    )
            else:
                # Fallback para resumo geral
                return await self._handle_summary_query(intent)

            return QueryResult(
                success=True,
                summary=summary,
                details=events,
                suggestions=[
                    "Refinar a busca",
                    "Ver resumo geral",
                ],
                metadata={},
            )

        return QueryResult(
            success=False,
            summary="Status store não disponível",
            details=[],
            suggestions=[],
            metadata={},
        )

    def _format_period(self, start: datetime, end: datetime) -> str:
        """Formata período para exibição."""
        now = datetime.now(timezone.utc)

        # Verifica se é hoje
        if start.date() == now.date():
            return "hoje"

        # Verifica se é ontem
        if start.date() == (now - timedelta(days=1)).date():
            return "ontem"

        # Período curto
        if (end - start).days <= 1:
            return f"em {start.strftime('%d/%m/%Y')}"

        return f"de {start.strftime('%d/%m')} a {end.strftime('%d/%m')}"


# =============================================================================
# API INTEGRATION
# =============================================================================


async def process_tws_query(query: str) -> QueryResult:
    """
    Processa uma query sobre o TWS.

    Função de conveniência para uso na API.

    Args:
        query: Pergunta em linguagem natural

    Returns:
        Resultado da query
    """
    from resync.core.tws_status_store import get_tws_status_store

    processor = TWSQueryProcessor(status_store=get_tws_status_store())
    return await processor.process_query(query)


# =============================================================================
# EXAMPLES
# =============================================================================

EXAMPLE_QUERIES = [
    "O que aconteceu ontem?",
    "Quais jobs falharam hoje?",
    "Tem algum padrão nas falhas?",
    "Histórico do job BATCH_PROCESS",
    "Como estão as workstations?",
    "Compara com a semana passada",
    "Jobs que falharam nas últimas 2 horas",
    "Problemas com WS001 esse mês",
]
