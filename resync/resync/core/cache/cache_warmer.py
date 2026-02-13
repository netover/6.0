"""
Cache Warming Service for Semantic Cache.

Pre-populates the semantic cache with frequent queries to
reduce latency and increase hit rate from startup.

Version: 5.3.18
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WarmingQuery:
    """Query for cache warming."""

    query: str
    category: str
    priority: int  # 1=alta, 2=média, 3=baixa
    expected_intent: str | None = None


@dataclass
class WarmingStats:
    """Warming statistics."""

    queries_processed: int = 0
    queries_cached: int = 0
    queries_skipped: int = 0
    errors: int = 0
    last_warm: datetime | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "queries_processed": self.queries_processed,
            "queries_cached": self.queries_cached,
            "queries_skipped": self.queries_skipped,
            "errors": self.errors,
            "last_warm": self.last_warm.isoformat() if self.last_warm else None,
            "duration_seconds": self.duration_seconds,
        }


class CacheWarmer:
    """
    Semantic cache warming service.

    Strategies:
    1. Predefined static queries (FAQ)
    2. Dynamic queries from history
    3. Critical job-based queries
    """

    # Most common static queries - TWS Operations
    STATIC_QUERIES: list[WarmingQuery] = [
        # Job Status (alta prioridade)
        WarmingQuery("qual o status do job", "job_status", 1, "job_details"),
        WarmingQuery("job está rodando", "job_status", 1, "job_details"),
        WarmingQuery("último run do job", "job_status", 1, "job_details"),
        WarmingQuery("próxima execução do job", "job_status", 1, "job_details"),
        WarmingQuery("job finalizou com sucesso", "job_status", 1, "job_details"),
        WarmingQuery("job terminou", "job_status", 1, "job_details"),
        # Troubleshooting (alta prioridade)
        WarmingQuery("job falhou o que fazer", "troubleshooting", 1, "troubleshooting"),
        WarmingQuery("como resolver RC 12", "troubleshooting", 1, "error_lookup"),
        WarmingQuery("job abendou", "troubleshooting", 1, "troubleshooting"),
        WarmingQuery("erro de conexão TWS", "troubleshooting", 1, "troubleshooting"),
        WarmingQuery("como reiniciar job", "troubleshooting", 1, "troubleshooting"),
        WarmingQuery("job travado", "troubleshooting", 1, "troubleshooting"),
        WarmingQuery("timeout no job", "troubleshooting", 1, "troubleshooting"),
        # Error Codes (alta prioridade)
        WarmingQuery("o que significa RC 8", "error_codes", 1, "error_lookup"),
        WarmingQuery("código de erro 12", "error_codes", 1, "error_lookup"),
        WarmingQuery("AWKR0001", "error_codes", 1, "error_lookup"),
        WarmingQuery("erro AWSBCT001I", "error_codes", 1, "error_lookup"),
        WarmingQuery("return code 4", "error_codes", 1, "error_lookup"),
        # Dependências (média prioridade)
        WarmingQuery("quais as dependências do job", "dependency", 2, "dependency_chain"),
        WarmingQuery("jobs predecessores", "dependency", 2, "dependency_chain"),
        WarmingQuery("cadeia de dependências", "dependency", 2, "dependency_chain"),
        WarmingQuery("jobs que rodam antes", "dependency", 2, "dependency_chain"),
        WarmingQuery("sequência de execução", "dependency", 2, "dependency_chain"),
        # Impacto (média prioridade)
        WarmingQuery("impacto se job falhar", "impact", 2, "impact_analysis"),
        WarmingQuery("quantos jobs afetados", "impact", 2, "impact_analysis"),
        WarmingQuery("análise de impacto", "impact", 2, "impact_analysis"),
        WarmingQuery("jobs dependentes", "impact", 2, "impact_analysis"),
        # Recursos (média prioridade)
        WarmingQuery("recursos do job", "resources", 2, "resource_conflict"),
        WarmingQuery("conflito de recursos", "resources", 2, "resource_conflict"),
        WarmingQuery("workstation do job", "resources", 2, "job_details"),
        # Documentação (baixa prioridade)
        WarmingQuery("documentação TWS", "documentation", 3, "documentation"),
        WarmingQuery("manual de operação", "documentation", 3, "documentation"),
        WarmingQuery("boas práticas TWS", "documentation", 3, "documentation"),
        WarmingQuery("como usar TWS", "documentation", 3, "documentation"),
        WarmingQuery("guia de referência", "documentation", 3, "documentation"),
        # Jobs Críticos (média prioridade)
        WarmingQuery("jobs críticos do dia", "critical", 2, "critical_jobs"),
        WarmingQuery("jobs prioritários", "critical", 2, "critical_jobs"),
        WarmingQuery("SLA críticos", "critical", 2, "critical_jobs"),
    ]

    def __init__(
        self,
        cache=None,
        retrieval_service=None,
        router=None,
        db_session=None,
    ):
        """
        Initializes the cache warmer.

        Args:
            cache: SemanticCache instance
            retrieval_service: UnifiedRetrievalService instance
            router: EmbeddingRouter instance
            db_session: Database session for historical queries
        """
        self.cache = cache
        self.retrieval = retrieval_service
        self.router = router
        self.db = db_session
        self._warming_in_progress = False
        self._stats = WarmingStats()

    async def warm_static_queries(self, priority: int = 3) -> int:
        """
        Warms cache with static queries.

        Args:
            priority: Máximo nível de prioridade a processar (1-3)

        Returns:
            Number of queries effectively cached
        """
        queries = [q for q in self.STATIC_QUERIES if q.priority <= priority]
        logger.info("Warming %s static queries (priority <= %s)", len(queries), priority)
        return await self._process_queries(queries)

    async def warm_critical_jobs(self, job_names: list[str] | None = None) -> int:
        """
        Warms cache with critical job queries.

        Args:
            job_names: Lista de nomes de jobs. Se None, usa lista default.

        Returns:
            Number of cached queries
        """
        # Default critical jobs (can be obtained from TWS in production)
        if job_names is None:
            job_names = [
                "BATCH_DIARIO",
                "FECHAMENTO_MES",
                "BACKUP_NOTURNO",
                "ETL_PRINCIPAL",
                "REPORT_REGULATORIO",
                "INTEGRACAO_SAP",
                "CARGA_DW",
                "RECONCILIACAO",
            ]

        warming_queries = []
        for job in job_names:
            warming_queries.extend(
                [
                    WarmingQuery(f"status do job {job}", "critical_job", 1),
                    WarmingQuery(f"dependências do job {job}", "critical_job", 1),
                    WarmingQuery(f"impacto se {job} falhar", "critical_job", 1),
                    WarmingQuery(f"próxima execução {job}", "critical_job", 2),
                ]
            )

        logger.info("Warming %s critical job queries", len(warming_queries))
        return await self._process_queries(warming_queries)

    def warm_from_history(
        self,
        days: int = 30,
        limit: int = 100,
    ) -> int:
        """
        Warms cache with most frequent historical queries.

        Args:
            days: Período em dias para análise
            limit: Número máximo de queries

        Returns:
            Número de queries cacheadas
        """
        if not self.db:
            logger.warning("Database session not available for history")
            return 0

        try:
            # Query para obter queries mais frequentes
            datetime.now(timezone.utc) - timedelta(days=days)

            # DEBT: Implement real DB query for cache warming history (low priority)
            # Retuning 0 for now
            logger.info("Warm from history: feature pending implementation")
            return 0

        except Exception as e:
            logger.error("Error getting history: %s", e)
            self._stats.errors += 1
            return 0

    async def _process_queries(self, queries: list[WarmingQuery]) -> int:
        """
        Processa lista de queries para warming.

        Args:
            queries: Lista de WarmingQuery

        Returns:
            Número de queries efetivamente cacheadas
        """
        cached_count = 0

        for wq in queries:
            self._stats.queries_processed += 1

            try:
                # Verificar se já está no cache
                if self.cache:
                    existing = await self.cache.get(wq.query)
                    if existing:
                        logger.debug("Query já em cache: %s...", wq.query[:40])
                        self._stats.queries_skipped += 1
                        continue

                # Classificar intent
                classification = None
                if self.router:
                    try:
                        classification = await self.router.classify(wq.query)
                    except Exception as e:
                        logger.debug("Router não disponível: %s", e)

                # Buscar resposta
                result = None
                if self.retrieval:
                    try:
                        intent = classification.intent if classification else None
                        result = await self.retrieval.retrieve(
                            query=wq.query,
                            intent=intent,
                        )
                    except Exception as e:
                        logger.debug("Retrieval não disponível: %s", e)

                # Armazenar no cache
                if self.cache and result:
                    response_data = {
                        "intent": classification.intent.value
                        if classification
                        else wq.expected_intent,
                        "confidence": classification.confidence if classification else 0.0,
                        "source": "cache_warmer",
                        "category": wq.category,
                        "priority": wq.priority,
                        "warmed_at": datetime.now(timezone.utc).isoformat(),
                    }

                    if hasattr(result, "documents"):
                        response_data["documents"] = [
                            d.model_dump() if hasattr(d, "model_dump") else d for d in result.documents
                        ]

                    if hasattr(result, "graph_data"):
                        response_data["graph_data"] = result.graph_data

                    await self.cache.set(
                        query=wq.query,
                        response=response_data,
                        metadata={
                            "source": "cache_warmer",
                            "category": wq.category,
                        },
                    )
                    cached_count += 1
                    self._stats.queries_cached += 1
                    logger.debug("Cached: %s...", wq.query[:40])
                else:
                    # v5.9.6: Fixed - don't count as cached when cache/retrieval unavailable
                    # Previously this was inflating metrics without actual caching
                    self._stats.queries_skipped += 1
                    logger.debug("Skipped (no cache available): %s...", wq.query[:40])

            except Exception as e:
                logger.error("Erro no warming de '%s...': %s", wq.query[:40], e)
                self._stats.errors += 1

        return cached_count

    async def full_warm(self, include_history: bool = False) -> dict[str, Any]:
        """
        Executa warming completo do cache.

        Ordem de execução:
        1. Queries estáticas (prioridade alta)
        2. Jobs críticos
        3. Queries estáticas (todas)
        4. Histórico (se habilitado)

        Args:
            include_history: Se deve incluir queries do histórico

        Returns:
            Estatísticas do warming
        """
        if self._warming_in_progress:
            return {"error": "Warming já em progresso", "stats": self._stats.to_dict()}

        self._warming_in_progress = True
        start_time = datetime.now(timezone.utc)

        try:
            results = {
                "static_high_priority": await self.warm_static_queries(priority=1),
                "critical_jobs": await self.warm_critical_jobs(),
                "static_all": await self.warm_static_queries(priority=3),
            }

            if include_history:
                results["historical"] = self.warm_from_history()

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._stats.last_warm = start_time
            self._stats.duration_seconds = duration

            results["total_cached"] = sum(results.values())
            results["duration_seconds"] = round(duration, 2)
            results["stats"] = self._stats.to_dict()

            logger.info(
                "Cache warming completo: {results['total_cached']} queries em {duration:.2f}s"
            )

            return results

        finally:
            self._warming_in_progress = False

    def get_stats(self) -> dict[str, Any]:
        """Retorna estatísticas do warming."""
        return self._stats.to_dict()

    def get_static_queries_count(self) -> dict[str, int]:
        """Retorna contagem de queries por prioridade."""
        counts = {"priority_1": 0, "priority_2": 0, "priority_3": 0, "total": 0}
        for q in self.STATIC_QUERIES:
            counts[f"priority_{q.priority}"] += 1
            counts["total"] += 1
        return counts

    @property
    def is_warming(self) -> bool:
        """Retorna se warming está em progresso."""
        return self._warming_in_progress


# Singleton instance
_warmer_instance: CacheWarmer | None = None


def get_cache_warmer(
    cache=None,
    retrieval_service=None,
    router=None,
    db_session=None,
) -> CacheWarmer:
    """
    Obtém instância singleton do CacheWarmer.

    Na primeira chamada, cria a instância com os serviços fornecidos.
    Chamadas subsequentes retornam a mesma instância.
    """
    global _warmer_instance

    if _warmer_instance is None:
        _warmer_instance = CacheWarmer(
            cache=cache,
            retrieval_service=retrieval_service,
            router=router,
            db_session=db_session,
        )

    return _warmer_instance


async def warm_cache_on_startup(priority: int = 1) -> dict[str, Any]:
    """
    Executa cache warming no startup da aplicação.

    Chamado durante o lifespan da aplicação FastAPI.
    Faz warming apenas de queries de alta prioridade para não
    atrasar muito o boot.

    Args:
        priority: Nível máximo de prioridade (1-3)

    Returns:
        Estatísticas do warming
    """
    try:
        warmer = get_cache_warmer()

        # Warming apenas queries de alta prioridade no startup
        result = await warmer.warm_static_queries(priority=priority)

        logger.info("Cache warming inicial: %s queries (priority <= %s)", result, priority)

        return {
            "queries_warmed": result,
            "priority": priority,
            "stats": warmer.get_stats(),
        }

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.warning("Cache warming falhou (não crítico): %s", e)
        return {
            "error": str(e),
            "queries_warmed": 0,
        }
