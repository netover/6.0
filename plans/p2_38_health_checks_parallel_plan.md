# Plano de Refatoração P2-38: Health Checks Paralelos

## Problema Atual

A função `get_system_health` em [`resync/api/routes/admin/main.py`](resync/api/routes/admin/main.py:983) executa verificações de saúde **sequencialmente**, causando latência acumulada:

```
Tempo total = T(database) + T(redis) + T(llm) + T(rag) + ...
            ≈ 5s + 5s + 5s + 5s = 20+ segundos
```

## Solução Proposta

Utilizar `asyncio.gather()` para executar verificações independentes **em paralelo**:

```
Tempo total = max(T(database), T(redis), T(llm), T(rag), ...)
            ≈ 5 segundos (paralelização máx)
```

---

## Arquitetura da Solução

### 1. Criar Funções de Health Check Independentes

Cada componente terá sua própria função async que retorna `ComponentHealth`:

```python
async def _check_database() -> ComponentHealth:
    """Verifica saúde do banco de dados."""
    try:
        start = time.perf_counter()
        from resync.core.context_store import ContextStore
        store = ContextStore()
        await store.initialize()
        latency = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            status="healthy",
            latency_ms=round(latency, 2),
            message="SQLite connected",
        )
    except Exception as e:
        return ComponentHealth(
            status="unhealthy",
            message=f"Database error: {str(e)[:100]}",
        )

async def _check_redis() -> ComponentHealth:
    """Verifica saúde do Redis."""
    # ... implementação similar
    pass

async def _check_llm() -> ComponentHealth:
    """Verifica saúde do LLM."""
    # ... implementação similar
    pass

async def _check_rag() -> ComponentHealth:
    """Verifica saúde do RAG."""
    # ... implementação similar
    pass
```

### 2. Executar em Paralelo com asyncio.gather

```python
async def get_system_health(request: Request) -> SystemHealthResponse:
    """Get comprehensive system health status."""
    
    # Executar todas as verificações em paralelo
    results = await asyncio.gather(
               _check_redis(),
        _ _check_database(),
check_llm(),
        _check_rag(),
        return_exceptions=True,  # Tratamento de exceções
    )
    
    # Mapear resultados
    component_names = ["database", "redis", "llm", "rag"]
    components = {}
    overall_healthy = True
    
    for name, result in zip(component_names, results):
        if isinstance(result, Exception):
            components[name] = ComponentHealth(
                status="unhealthy",
                message=f"Check failed: {str(result)[:100]}",
            )
            overall_healthy = False
        else:
            components[name] = result
            if result.status == "unhealthy":
                overall_healthy = False
    
    # ... resto da função
```

---

## Ordem de Implementação

### Fase 1: Extrair Funções de Check

1. **Criar `_check_database()`**
   - Isolar lógica de verificação do banco
   - Retornar `ComponentHealth` diretamente

2. **Criar `_check_redis()`**
   - Isolar lógica de verificação do Redis
   - Tratar caso Redis não configurado

3. **Criar `_check_llm()`**
   - Isolar verificação HTTP do LLM
   - Timeout configurável

4. **Criar `_check_rag()`**
   - Isolar verificação do RAG/pgvector
   - Fallback para RAG_SERVICE_URL

### Fase 2: Refatorar get_system_health

1. Substituir blocos sequenciais por `asyncio.gather()`
2. Tratar exceções com `return_exceptions=True`
3. Agregar resultados

### Fase 3: Testes

1. Testar resposta rápida (paralelo)
2. Testar degradação graciosa (se um check falha, outros continuam)
3. Testar timeout overall

---

## Benefícios

| Métrica | Antes | Depois |
|---------|-------|--------|
| Latência Máxima | ~20s | ~5s |
|throughput requests/min | 3 | 12 |
| Bloqueio de API | Alto | Mínimo |

---

## Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Exceções não tratadas | Usar `return_exceptions=True` |
| Recursos não limpos | Context managers em cada função |
| Timeout muito longo | Timeout individual em cada check |

---

## Código Exemplificado Completo

```python
async def _check_database() -> ComponentHealth:
    """Check database health."""
    import time
    try:
        start = time.perf_counter()
        from resync.core.context_store import ContextStore
        store = ContextStore()
        await store.initialize()
        latency = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            status="healthy",
            latency_ms=round(latency, 2),
            message="SQLite connected",
        )
    except Exception as e:
        return ComponentHealth(
            status="unhealthy",
            message=f"Database error: {str(e)[:100]}",
        )

async def get_system_health(request: Request) -> SystemHealthResponse:
    """Get comprehensive system health status."""
    import asyncio
    
    # P2-38 FIX: Run health checks in parallel
    results = await asyncio.gather(
        _check_database(),
        _check_redis(),
        _check_llm(),
        _check_rag(),
        return_exceptions=True,
    )
    
    # Aggregate results
    components = {
        "database": results[0] if not isinstance(results[0], Exception) 
                    else ComponentHealth(status="unhealthy", message="Check failed"),
        "redis": results[1] if not isinstance(results[1], Exception)
                 else ComponentHealth(status="unhealthy", message="Check failed"),
        "llm": results[2] if not isinstance(results[2], Exception)
               else ComponentHealth(status="unhealthy", message="Check failed"),
        "rag": results[3] if not isinstance(results[3], Exception)
              else ComponentHealth(status="unhealthy", message="Check failed"),
    }
    
    overall_healthy = all(
        c.status == "healthy" for c in components.values()
    )
    
    return SystemHealthResponse(...)
```

---

## Checklist de Implementação

- [ ] Extrair `_check_database()` para função independente
- [ ] Extrair `_check_redis()` para função independente
- [ ] Extrair `_check_llm()` para função independente
- [ ] Extrair `_check_rag()` para função independente
- [ ] Refatorar `get_system_health()` para usar `asyncio.gather()`
- [ ] Adicionar testes de carga para verificar paralelização
- [ ] Verificar que exceções em um check não afeta outros
