# Plano de Correção: task_tracker.py v2

## Visão Geral

Este documento detalha o plano para aplicar as correções de hardenização no módulo `resync/core/task_tracker.py`, abordando os problemas P0, P1 e P2 identificados na análise.

---

## Problemas a Corrigir

### P0 - Críticos

| ID | Problema | Solução |
|----|----------|---------|
| P0-1 | `asyncio.create_task()` falha sem loop em execução | Validar loop explicitamente com fail-fast |
| P0-1 | Criação ilimitada de tasks (DoS) | Adicionar hard limit de 10,000 tasks |
| P0-2 | `cancel_all_tasks()` pode travar após timeout | Two-phase shutdown com timeout |

### P1 - Alto

| ID | Problema | Solução |
|----|----------|---------|
| P1-1 | `threading.Lock` pode bloquear event loop | Documentar decisão (manter para cross-thread) |
| P1-2 | `wait_for_tasks()` cancela silenciosamente | Adicionar parâmetro `cancel_pending=False` |
| P1-3 | Estatísticas enganosas | Separar `timed_out`, `stuck`, `cancelled` |

### P2 - Manutenibilidade

| ID | Problema | Solução |
|----|----------|---------|
| P2 | Exceções no callback não tratadas | Try/except robusto |
| P2 | Timeout negativo/inválido | Validar timeout > 0 |

---

## Plano de Execução

### Fase 1: Preparação

1. [ ] Ler o estado atual do arquivo `resync/core/task_tracker.py`
2. [ ] Identificar linhas exatas para cada modificação
3. [ ] Criar backup do arquivo original

### Fase 2: Correções P0

#### P0-1: Validação de Loop e Limite

**Modificar `create_tracked_task()`:**

```python
def create_tracked_task(
    coro: Coroutine[Any, Any, T],
    name: str | None = None,
    *,
    cancel_on_shutdown: bool = True,
) -> asyncio.Task[T]:
    # Validação de loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError as e:
        raise RuntimeError(
            "create_tracked_task() must be called with a running event loop "
            "(e.g., inside FastAPI lifespan/request/websocket)."
        ) from e

    task: asyncio.Task[T] = loop.create_task(coro, name=name)

    if cancel_on_shutdown:
        with _tasks_lock:
            # Hard limit para evitar DoS
            if len(_background_tasks) >= 10_000:
                raise RuntimeError(
                    "Too many background tasks tracked (limit=10000)."
                )
            _background_tasks.add(task)
            total_tasks = len(_background_tasks)
    else:
        with _tasks_lock:
            total_tasks = len(_background_tasks)
    
    # ... resto do código
```

#### P0-2: Two-Phase Shutdown

**Modificar `cancel_all_tasks()`:**

```python
async def cancel_all_tasks(timeout: float = 5.0) -> dict[str, int]:
    # Validação de timeout
    if timeout <= 0:
        raise ValueError("timeout must be > 0")

    with _tasks_lock:
        tasks_to_cancel = list(_background_tasks)

    total = len(tasks_to_cancel)
    if total == 0:
        return {
            "total": 0, "cancelled": 0, "completed": 0, 
            "errors": 0, "timed_out": 0, "stuck": 0
        }

    logger.info("Cancelling %s background tasks...", total)

    # Fase 1: Cancelar e esperar
    for t in tasks_to_cancel:
        t.cancel()

    done, pending = await asyncio.wait(tasks_to_cancel, timeout=timeout)

    # Processar tasks completadas
    cancelled = 0
    completed = 0
    errors = 0

    for t in done:
        if t.cancelled():
            cancelled += 1
            continue
        exc = t.exception()
        if exc is not None:
            logger.error(
                "background_task_error",
                extra={"task": t.get_name(), "error": str(exc)},
                exc_info=exc
            )
            errors += 1
        else:
            completed += 1

    timed_out = len(pending)
    stuck = 0

    # Fase 2: Se ainda há pending, segunda chance com timeout
    if pending:
        logger.warning(
            "background_tasks_timeout",
            extra={
                "pending_count": len(pending),
                "tasks": [t.get_name() for t in pending]
            },
        )

        for t in pending:
            t.cancel()

        done2, pending2 = await asyncio.wait(pending, timeout=timeout)

        # Processar results da fase 2
        for t in done2:
            if t.cancelled():
                cancelled += 1
            else:
                exc = t.exception()
                if exc is not None:
                    logger.error(
                        "background_task_error",
                        extra={"task": t.get_name(), "error": str(exc)},
                        exc_info=exc
                    )
                    errors += 1
                else:
                    completed += 1

        if pending2:
            stuck = len(pending2)
            logger.error(
                "background_tasks_stuck_after_cancel",
                extra={
                    "stuck_count": stuck,
                    "tasks": [t.get_name() for t in pending2]
                },
            )
            # IMPORTANTE: Manter em _background_tasks para visibilidade

    stats = {
        "total": total,
        "cancelled": cancelled,
        "completed": completed,
        "errors": errors,
        "timed_out": timed_out,
        "stuck": stuck,
    }
    logger.info("Background tasks shutdown complete", extra=stats)
    return stats
```

### Fase 3: Correções P1

#### P1-2: Parâmetro cancel_pending

**Modificar `wait_for_tasks()`:**

```python
async def wait_for_tasks(
    timeout: float | None = None,
    *,
    cancel_pending: bool = False  # Não cancelar por padrão
) -> bool:
    # Validar timeout
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be > 0")

    with _tasks_lock:
        tasks = set(_background_tasks)

    if not tasks:
        return True

    done, pending = await asyncio.wait(tasks, timeout=timeout)

    if pending and cancel_pending:
        for t in pending:
            t.cancel()
        await asyncio.wait(pending, timeout=timeout)

    return len(pending) == 0
```

### Fase 4: Correções P2

#### P2: Robustez no Callback

**Modificar o done callback:**

```python
def _remove_and_log(t: asyncio.Task[Any]) -> None:
    try:
        with _tasks_lock:
            _background_tasks.discard(t)

        if not t.cancelled():
            exc = t.exception()
            if exc is not None:
                logger.error(
                    "unhandled_background_task_error",
                    extra={"task": t.get_name(), "error": str(exc)},
                    exc_info=exc,
                )
    except Exception:
        logger.exception(
            "task_tracker_done_callback_failed",
            extra={"task": t.get_name()}
        )
```

---

## Checklist de Validação

### Antes de Aplicar

- [ ] Backup do arquivo original
- [ ] Todos os testes existentes passando

### Após Aplicar

- [ ] Tests de smoke passing
- [ ] Sem `RuntimeError` em contextos síncronos
- [ ] Two-phase shutdown funciona corretamente
- [ ] Estatísticas retorn valores corretos
- [ ] `amwait_for_tasks` não cancela por padrão

---

## Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Breaking change na API | Manter compatibilidade retroativa |
| Tests quebrando | Atualizar testes para novos retornos |
| Performance com muitos tasks | Limite de 10k previne DoS |

---

## Referência

Ver análise completa em: `analise_bugs_erros.txt`
