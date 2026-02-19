Auditoria completa (estática) dos arquivos Python — Projeto 6.0-new1-local
Gerado em: 2026-02-19T00:10:45.045886Z

Escopo
- Arquivo analisado: 6.0-new1-local (1).zip
- Diretório extraído: /mnt/data/audit_6.0_new1_local
- .py encontrados no ZIP (excluindo '.minimax'): 531
- Escopo considerado (excluindo venv/site-packages): 531
- Observação: esta auditoria é estática/heurística (AST + regex + compileall). Para 100% de cobertura funcional, rode testes e a checklist arquivo-a-arquivo.

Resultado do compileall
- Return code: 1
- Saída (primeiros erros):
  *** Error compiling '/mnt/data/audit_6.0_new1_local/6.0-new1-local/resync/scripts/install_postgres.py'...
    File "/mnt/data/audit_6.0_new1_local/6.0-new1-local/resync/scripts/install_postgres.py", line 651
      global DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  SyntaxError: name 'DB_NAME' is used prior to global declaration

  *** Error compiling '/mnt/data/audit_6.0_new1_local/6.0-new1-local/resync/scripts/install_redis.py'...
    File "/mnt/data/audit_6.0_new1_local/6.0-new1-local/resync/scripts/install_redis.py", line 316
      global REDIS_HOST, REDIS_PORT, REDIS_PASSWORD
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  SyntaxError: name 'REDIS_HOST' is used prior to global declaration

Sumário de achados (contagem)
- logging_kwargs_std_logger: 230
- await_inside_lock: 51
- eval_or_exec: 3
- module_package_collision: 2
- hardcoded_api_key: 1
- hardcoded_password: 1
- httpx_without_timeout: 1
- requests_without_timeout: 1

Colisões de módulo/package detectadas
- /mnt/data/audit_6.0_new1_local/6.0-new1-local/resync/api: 'auth.py' e 'auth/' coexistem -> risco de import ambíguo
- /mnt/data/audit_6.0_new1_local/6.0-new1-local/resync/core: 'metrics.py' e 'metrics/' coexistem -> risco de import ambíguo

Detalhamento por arquivo (com linha aproximada e proposta de correção)
================================================================================

Arquivo: /mnt/data/audit_6.0_new1_local/6.0-new1-local/resync/api
  - [module_package_collision] linha 0: Both file 'auth.py' and package dir 'auth/' exist under same parent
    Correção sugerida: Renomear arquivo ou package para evitar colisão; atualizar imports e testes.

Arquivo: /mnt/data/audit_6.0_new1_local/6.0-new1-local/resync/core
  - [module_package_collision] linha 0: Both file 'metrics.py' and package dir 'metrics/' exist under same parent
    Correção sugerida: Renomear arquivo ou package para evitar colisão; atualizar imports e testes.

Arquivo: 6.0-new1-local/resync/api/auth_legacy.py
  - [await_inside_lock] linha 103: Await inside async-with lock context: self._get_lockout_lock()
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [logging_kwargs_std_logger] linha 105: logger.warning(
                    "Authentication attempt from locked out IP",
                    extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 141: logger.warning(
                "Failed authentication attempt",
                extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 157: logger.info(
            "Successful authentication",
            extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/chat.py
  - [logging_kwargs_std_logger] linha 97: logger.warning("Failed to send error message due to an unexpected error.", exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 108: logger.error("IA Auditor timed out during execution.", exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 110: logger.error("IA Auditor encountered a knowledge graph error.", exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 112: logger.error("IA Auditor encountered a database error.", exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 114: logger.error("IA Auditor encountered an audit-specific error.", exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 122: logger.critical(
            "IA Auditor background task failed with an unhandled exception.",
            exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 255: logger.error("Error in agent interaction: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 362: logger.error(
            "Agent-related error in WebSocket for agent '%s': %s", agent_id, exc, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 372: logger.critical("Unhandled exception in WebSocket for agent '%s'", agent_id, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/core/security.py
  - [logging_kwargs_std_logger] linha 33: logger.error("password_verification_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/dependencies_v2.py
  - [await_inside_lock] linha 118: Await inside async-with lock context: _get_lock("tws")
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/api/enhanced_endpoints.py
  - [logging_kwargs_std_logger] linha 133: logger.error("Error investigating job %s: %s", job_name, e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 166: logger.error("Error checking system health: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 212: logger.error("Error getting failed jobs: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 293: logger.error("Error generating job summary: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/middleware/cors_config.py
  - [logging_kwargs_std_logger] linha 206: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 242: logger.debug("suppressed_exception", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/middleware/cors_monitoring.py
  - [logging_kwargs_std_logger] linha 267: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/middleware/database_security_middleware.py
  - [logging_kwargs_std_logger] linha 130: logger.warning(
                    "sql_injection_blocked",
                    key=key,
                    value_preview=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 187: logger.debug("failed_to_extract_request_body", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 239: logger.error("failed_to_log_request_outcome", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/middleware/error_handler.py
  - [logging_kwargs_std_logger] linha 67: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/routes/admin/config.py
  - [logging_kwargs_std_logger] linha 137: logger.error("config_load_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 370: logger.error("backup_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 442: logger.error("audit_query_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/routes/admin/main.py
  - [logging_kwargs_std_logger] linha 114: logger.error("Failed to render admin dashboard: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 142: logger.error("Failed to render API keys admin page: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 207: logger.error("Failed to get admin configuration: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 292: logger.error("Failed to update Teams configuration: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 321: logger.error("Failed to get Teams health status: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 375: logger.error("Failed to send test Teams notification: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 432: logger.error("Failed to get admin status: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 525: logger.error("Failed to update TWS configuration: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 578: logger.error("Failed to update system configuration: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 645: logger.error("Failed to retrieve logs: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 700: logger.error("Failed to clear cache: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 741: logger.error("Failed to create backup: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 785: logger.error("Failed to list backups: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 844: logger.error("Failed to restore backup: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 1174: logger.error("Failed to get audit logs: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/routes/admin/prompts.py
  - [logging_kwargs_std_logger] linha 170: logger.error("list_prompts_error", exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/routes/admin/semantic_cache.py
  - [logging_kwargs_std_logger] linha 487: logger.debug("suppressed_exception", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/routes/admin/v2.py
  - [logging_kwargs_std_logger] linha 214: logger.warning("Redis health check failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 247: logger.warning("Database health check failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 282: logger.warning("TWS health check failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 319: logger.warning("LLM health check failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 365: logger.warning("Failed to get circuit breaker health", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 374: logger.warning("Failed to get redis strategy", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 423: logger.warning("Failed to get breaker %s", name, error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 677: logger.error("Reindex failed", job_id=job_id, error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 925: logger.error("request_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/routes/cache.py
  - [logging_kwargs_std_logger] linha 417: logger.error("Error during cache invalidation: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/routes/core/auth.py
  - [eval_or_exec] linha 296: eval(
    Correção sugerida: Remover `eval/exec` ou isolar em ambiente seguro; validar/sanitizar entrada; preferir parsing seguro.

Arquivo: 6.0-new1-local/resync/api/routes/core/health.py
  - [logging_kwargs_std_logger] linha 153: logger.warning("Failed to increment health check metrics: %s", metrics_e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 475: logger.error("Component recovery failed: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 529: logger.warning(
                "Redis health check failed - idempotency may be compromised",
                extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 551: logger.error("Error checking Redis health: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 630: logger.error("Error during health service shutdown: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/routes/monitoring/metrics_dashboard.py
  - [logging_kwargs_std_logger] linha 240: logger.warning("psutil_process_metrics_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 327: logger.warning("psutil_net_connections_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/routes/rag/query.py
  - [logging_kwargs_std_logger] linha 57: logger.error("request_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 63: logger.error("file_validation_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/routes/rag/upload.py
  - [logging_kwargs_std_logger] linha 86: logger.info(
            "rag_document_uploaded",
            user=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 103: logger.error("File processing error: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 109: logger.error("Failed to process uploaded file: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/services/rag_config.py
  - [logging_kwargs_std_logger] linha 34: logger.warning(
                "insecure_database_url_detected",
                hint=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/utils/helpers.py
  - [logging_kwargs_std_logger] linha 114: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/api/validation/auth.py
  - [hardcoded_api_key] linha 26: API_KEY = "api_key"
    Correção sugerida: Remover segredo do código; usar env vars/secret manager; invalidar/rotacionar a chave.

Arquivo: 6.0-new1-local/resync/config/security.py
  - [logging_kwargs_std_logger] linha 48: logger.info(
        "security_headers_configured",
        enforce_https=settings.is_production,
        environment=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/__init__.py
  - [logging_kwargs_std_logger] linha 59: logger.debug("suppressed_exception", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 315: logger.debug("suppressed_exception", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/agent_manager.py
  - [await_inside_lock] linha 478: Await inside async-with lock context: self._get_agent_lock()
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/anomaly_detector.py
  - [await_inside_lock] linha 566: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/audit_db.py
  - [logging_kwargs_std_logger] linha 65: logger.warning("auditdb_get_records_sync_shim_used", limit=limit, offset=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/audit_lock.py
  - [await_inside_lock] linha 346: Await inside async-with lock context: await lock.acquire(memory_id, timeout)
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [eval_or_exec] linha 297: eval(
    Correção sugerida: Remover `eval/exec` ou isolar em ambiente seguro; validar/sanitizar entrada; preferir parsing seguro.
  - [logging_kwargs_std_logger] linha 47: logger.info("DistributedAuditLock initialized with Redis", redis_url=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 144: logger.warning("forcefully_released_audit_lock_for_memory", memory_id=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 201: logger.critical(
                "Unexpected critical error cleaning up expired audit locks.",
                exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 300: logger.debug("successfully_released_audit_lock", lock_key=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 318: logger.error("Unexpected error during lock release", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 319: logger.error(
                "lock_details",
                key=self.lock_key,
                value=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/audit_to_kg_pipeline.py
  - [await_inside_lock] linha 204: Await inside async-with lock context: self.processing_lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/cache/advanced_cache.py
  - [await_inside_lock] linha 399: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 548: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/cache/async_cache.py
  - [logging_kwargs_std_logger] linha 179: logger.info(
            "AsyncTTLCache initialized",
            extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/cache/cache_factory.py
  - [await_inside_lock] linha 148: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/cache/cache_hierarchy.py
  - [logging_kwargs_std_logger] linha 195: logger.warning(
                    "encryption_failed",
                    error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 213: logger.warning(
                    "decryption_failed",
                    error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 272: logger.debug("cache_hierarchy_set", key=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/cache/cache_with_stampede_protection.py
  - [await_inside_lock] linha 75: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/cache/memory_manager.py
  - [logging_kwargs_std_logger] linha 235: logger.warning(
                    "Cache memory usage {estimated_memory_mb:.1f}MB approaching limit of {self.max_memory_mb}MB",
                    extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 257: logger.warning(
                        "Estimated cache memory usage {estimated_memory_mb:.1f}MB exceeds {self.max_memory_mb}MB limit",
                        extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 408: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 522: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/cache/query_cache.py
  - [await_inside_lock] linha 325: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 352: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/cache/redis_config.py
  - [logging_kwargs_std_logger] linha 226: logger.debug("suppressed_exception", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/cache/semantic_cache.py
  - [await_inside_lock] linha 445: Await inside async-with lock context: _get_cache_lock()
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/cache_utils.py
  - [logging_kwargs_std_logger] linha 158: logger.error("Failed to warm cache key %s: %s", key, e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 199: logger.error("Failed to invalidate pattern %s: %s", pattern, e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 317: logger.error("Failed to warm cache on startup: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/chaos_engineering.py
  - [logging_kwargs_std_logger] linha 109: logger.error(
                    "chaos_suite_timeout",
                    extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 202: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 269: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 299: logger.debug("suppressed_exception", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 349: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 361: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 373: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 434: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 500: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 567: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 583: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 589: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 723: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 803: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 863: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 872: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 923: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 1004: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 1057: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 1091: logger.warning(
            "⚠️ ChaosEngineer initialized - ensure this is NOT production!",
            extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 1110: logger.warning(
            "⚠️ FuzzingEngine initialized - ensure this is NOT production!",
            extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/config_hot_reload.py
  - [await_inside_lock] linha 156: Await inside async-with lock context: self._async_lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 223: Await inside async-with lock context: self._async_lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 254: Await inside async-with lock context: self._async_lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 298: Await inside async-with lock context: self._async_lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [logging_kwargs_std_logger] linha 124: logger.info("ConfigManager started", extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/config_persistence.py
  - [logging_kwargs_std_logger] linha 93: logger.error("Failed to load configuration: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 115: logger.error("Failed to save configuration: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 263: logger.error("Failed to restore backup: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/connection_manager.py
  - [logging_kwargs_std_logger] linha 81: logger.info(
                "broadcast_completed",
                successful_sends=successful_sends,
                message=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 93: logger.info("broadcasting_message", client_count=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 112: logger.error("Unexpected error during broadcast.", exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 122: logger.info(
                "json_broadcast_completed",
                successful_sends=successful_sends,
                message=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 150: logger.error("JSON serialization error during broadcast: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 152: logger.error("Unexpected error during JSON broadcast.", exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/database/engine.py
  - [logging_kwargs_std_logger] linha 104: logger.info(
            "Creating PostgreSQL database engine",
            extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/database_security.py
  - [logging_kwargs_std_logger] linha 358: logger.info("database_operation_audited", extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 379: logger.warning("database_security_violation", extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/di_container.py
  - [await_inside_lock] linha 190: Await inside async-with lock context: self._locks[interface]
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/document_graphrag.py
  - [logging_kwargs_std_logger] linha 109: logger.warning("document_graphrag_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/encoding_utils.py
  - [logging_kwargs_std_logger] linha 22: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/encrypted_audit.py
  - [await_inside_lock] linha 747: Await inside async-with lock context: self._fs_lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 748: Await inside async-with lock context: aiofiles.open(block_file, "w", encoding="utf-8")
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 757: Await inside async-with lock context: self._fs_lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/factories/redis_factory.py
  - [await_inside_lock] linha 265: Await inside async-with lock context: _get_async_lock()
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/file_ingestor.py
  - [logging_kwargs_std_logger] linha 62: logger.info("file_saved_successfully", path=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 65: logger.error("failed_to_save_file", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 73: logger.info("starting_file_ingestion", path=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 78: logger.error("conversion_failed", error=result.error, path=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 95: logger.info("file_ingested_successfully", path=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 98: logger.error("ingestion_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 110: logger.error("file_ingestor_shutdown_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/health/health_monitoring_coordinator.py
  - [await_inside_lock] linha 46: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 56: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/health/health_service_facade.py
  - [await_inside_lock] linha 86: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 107: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 129: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 354: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/ia_auditor.py
  - [await_inside_lock] linha 250: Await inside async-with lock context: await audit_lock.acquire(memory_id, timeout=30)
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/langfuse/prompt_manager.py
  - [await_inside_lock] linha 225: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/langgraph/checkpointer.py
  - [await_inside_lock] linha 126: Await inside async-with lock context: _get_checkpointer_lock()
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/litellm_init.py
  - [logging_kwargs_std_logger] linha 111: logger.warning("LiteLLM not installed: %s", import_err, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 150: logger.info(
                "Cost calculation unavailable (litellm not installed?): %s",
                import_err,
                exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 160: logger.warning("Could not calculate completion cost: %s", err, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/llm_optimizer.py
  - [logging_kwargs_std_logger] linha 315: logger.error("Error in LLM streaming", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/logging_utils.py
  - [logging_kwargs_std_logger] linha 183: logger.debug("suppressed_exception", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/performance_optimizer.py
  - [await_inside_lock] linha 369: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/redis_init.py
  - [await_inside_lock] linha 204: Await inside async-with lock context: self.lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [eval_or_exec] linha 288: eval(
    Correção sugerida: Remover `eval/exec` ou isolar em ambiente seguro; validar/sanitizar entrada; preferir parsing seguro.
  - [logging_kwargs_std_logger] linha 232: logger.debug("redis_client_close_failed", exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 260: logger.info(
                            "Redis initialized successfully",
                            extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 355: logger.error("Redis health check failed - connection may be lost", exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 359: logger.error("Unexpected error in Redis health check: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/resource_manager.py
  - [await_inside_lock] linha 259: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 293: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/smart_pooling.py
  - [await_inside_lock] linha 260: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/structured_logger.py
  - [logging_kwargs_std_logger] linha 797: logger.debug("suppressed_exception", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/task_tracker.py
  - [logging_kwargs_std_logger] linha 114: logger.debug(
        "Created tracked task",
        extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 193: logger.error("background_task_error", task=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 201: logger.warning(
            "background_tasks_timeout",
            pending_count=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 214: logger.info("Background tasks shutdown complete", extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 257: logger.debug(
        "Created tracked task (sync)",
        extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/teams_integration.py
  - [await_inside_lock] linha 125: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 222: Await inside async-with lock context: self._session_lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/utils/async_bridge.py
  - [logging_kwargs_std_logger] linha 75: logger.warning("background_task_failed", task=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 78: logger.debug("suppressed_exception", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/utils/common_error_handlers.py
  - [logging_kwargs_std_logger] linha 64: logger.error("%s: %s", error_message, e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 92: logger.error("%s: %s", error_message, e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/utils/json_commands.py
  - [logging_kwargs_std_logger] linha 78: logger.warning("JSON decode error", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/core/utils/secret_scrubber.py
  - [hardcoded_password] linha 193: password="***MASKED***"
    Correção sugerida: Remover credenciais hardcoded; usar env vars/secret manager; rotacionar se aplicável.

Arquivo: 6.0-new1-local/resync/core/websocket_pool_manager.py
  - [await_inside_lock] linha 83: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 103: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 135: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 250: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/core/write_ahead_log.py
  - [await_inside_lock] linha 183: Await inside async-with lock context: self.lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/knowledge/ingestion/chunking_eval.py
  - [logging_kwargs_std_logger] linha 499: logger.error("eval_retrieve_failed", query_id=query_id, error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 631: logger.info("eval_report_saved", path=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/knowledge/ingestion/document_converter.py
  - [logging_kwargs_std_logger] linha 288: logger.info("docling_convert_start", file=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 298: logger.info(
                "docling_convert_done",
                file=file_path.name,
                pages=result.pages,
                tables=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 306: logger.warning(
                "docling_convert_failed",
                file=file_path.name,
                error=result.error,
                time_s=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 391: logger.warning(
                "docling_subprocess_crashed",
                exitcode=proc.exitcode,
                file=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/knowledge/ingestion/embedding_service.py
  - [logging_kwargs_std_logger] linha 188: logger.info(
            "MultiProviderEmbeddingService initialized",
            extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/knowledge/ingestion/embeddings.py
  - [logging_kwargs_std_logger] linha 158: logger.warning(
                        "embedding_timeout",
                        attempt=attempt + 1,
                        batch_size=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 168: logger.error(
                        "embedding_error",
                        error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 177: logger.debug(
            "embeddings_generated",
            count=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 263: logger.info("mock_embedding_provider_initialized", dimension=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 275: logger.info(
        "litellm_embedding_provider_initialized",
        model=config.model,
        dimension=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/knowledge/ingestion/ingest.py
  - [logging_kwargs_std_logger] linha 353: logger.info(
                    "multi_view_generated",
                    doc_id=doc_id,
                    views_generated=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 359: logger.warning("multi_view_indexing_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 397: logger.info(
                        "document_kg_extracted",
                        doc_id=doc_id,
                        concepts=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 404: logger.warning("document_kg_extraction_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/knowledge/ingestion/pipeline.py
  - [logging_kwargs_std_logger] linha 184: logger.error("ingestion_failed", doc_id=doc_id, error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 202: logger.info(
            "document_ingested",
            doc_id=doc_id,
            source=source,
            format=converted.format,
            pages=converted.pages,
            tables=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/knowledge/kg_extraction/extractor.py
  - [logging_kwargs_std_logger] linha 99: logger.warning("kg_extract_concepts_failed", extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 119: logger.warning("kg_extract_edges_failed", extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/knowledge/kg_store/store.py
  - [logging_kwargs_std_logger] linha 217: logger.info(
            "kg_extraction_persisted",
            doc_id=doc_id,
            nodes=n_nodes,
            edges=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/knowledge/retrieval/cache_manager.py
  - [await_inside_lock] linha 186: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/knowledge/retrieval/hybrid_retriever.py
  - [logging_kwargs_std_logger] linha 353: logger.info(
                        "bm25_index_saved",
                        path=path,
                        size_bytes=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 363: logger.error("bm25_index_save_timeout", path=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 367: logger.error("bm25_index_save_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 382: logger.info("bm25_index_not_found_will_build", path=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 401: logger.info(
                        "bm25_index_loaded",
                        path=path,
                        size_bytes=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 411: logger.warning("bm25_index_locked_using_empty", path=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 415: logger.warning(
                "bm25_index_corrupted_rebuilding",
                error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 426: logger.error("bm25_index_cleanup_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 431: logger.error(
                "bm25_index_oom_using_empty",
                path=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 438: logger.warning(
                "bm25_index_load_failed_unknown",
                error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 843: logger.info("Attempting to load persisted BM25 index", path=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 851: logger.info(
                    "BM25 index loaded from disk",
                    num_docs=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 858: logger.warning(
                "BM25 index load failed, will rebuild",
                error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/knowledge/store/pgvector_store.py
  - [logging_kwargs_std_logger] linha 147: logger.debug("batch_upserted", collection=col, count=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 235: logger.debug(
            "query_completed",
            collection=col,
            results=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/services/advanced_graph_queries.py
  - [logging_kwargs_std_logger] linha 182: logger.debug(
            "temporal_state_recorded",
            entity_id=entity_id,
            timestamp=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 213: logger.debug(
                    "temporal_lookup_hit",
                    entity_id=entity_id,
                    query_time=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 221: logger.debug(
            "temporal_lookup_miss",
            entity_id=entity_id,
            query_time=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 331: logger.info(
                "conflict_resolved_by_temporal",
                entity_id=entity_id,
                resolved_timestamp=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 447: logger.info(
            "negation_query_executed",
            target=resource_or_job,
            total_jobs=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 734: logger.info(
            "intersection_analysis",
            entity_a=entity_a,
            entity_b=entity_b,
            common_predecessors=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 851: logger.debug(
            "explicit_edge_registered",
            source=source,
            target=target,
            relation=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 881: logger.debug(
            "co_occurrence_registered",
            entities=key,
            count=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 1004: logger.info(
            "graph_filtered_by_confidence",
            original_edges=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/services/llm_fallback.py
  - [requests_without_timeout] linha 547: requests.get(current_model, 0)
    Correção sugerida: Adicionar `timeout=` e lidar com exceptions; preferir sessões reutilizáveis.

Arquivo: 6.0-new1-local/resync/services/llm_service.py
  - [logging_kwargs_std_logger] linha 183: logger.info(
                "llm_resilience_configured",
                timeout_s=self._timeout_s,
                max_concurrency=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 201: logger.error("Failed to initialize LLM service (OpenAI error): %s", exc, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 213: logger.error("Failed to initialize LLM service: %s", exc, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 464: logger.error("Error in generate_response_with_tools: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 546: logger.error("Error generating LLM response: %s", exc, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 559: logger.error("Unexpected error generating LLM response: %s", exc, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 616: logger.error("Error generating streaming LLM response: %s", exc, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 629: logger.error(
                "Unexpected error generating streaming LLM response: %s",
                exc,
                exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 684: logger.debug(
                        "prompt_loaded_from_manager",
                        prompt_id=prompt.id,
                        agent_id=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 693: logger.warning("prompt_manager_fallback", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 780: logger.debug("rag_prompt_loaded_from_manager", prompt_id=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 785: logger.warning("rag_prompt_manager_fallback", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 815: logger.warning(
                "self_rag_hallucination_detected",
                query=query[:50],
                reflection=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 831: logger.info("self_rag_regenerated", query=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 833: logger.error("self_rag_regeneration_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 836: logger.debug("self_rag_grounded", query=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 904: logger.warning("hallucination_check_failed", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 957: logger.error("exception_caught", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/services/tws_cache.py
  - [await_inside_lock] linha 315: Await inside async-with lock context: lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 362: Await inside async-with lock context: lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/services/tws_unified.py
  - [await_inside_lock] linha 215: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 270: Await inside async-with lock context: self._lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 494: Await inside async-with lock context: _get_tws_client_lock()
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.
  - [await_inside_lock] linha 506: Await inside async-with lock context: _get_tws_client_lock()
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

Arquivo: 6.0-new1-local/resync/tests/conftest.py
  - [httpx_without_timeout] linha 287: httpx.get("https://api.example.com/data")
    Correção sugerida: Adicionar timeout explícito e configurar client reutilizável/limites.
  - [logging_kwargs_std_logger] linha 112: logger.debug("suppressed_exception", error=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/tools/definitions/tws.py
  - [logging_kwargs_std_logger] linha 59: logger.error("TWS connection error in TWSStatusTool: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 64: logger.error("Value error in TWSStatusTool: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 67: logger.error("Unexpected error in TWSStatusTool: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 108: logger.error("TWS connection error in TWSTroubleshootingTool: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 113: logger.error(
                "Data or processing error in TWSTroubleshootingTool: %s",
                e,
                exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 120: logger.error("Unexpected error in TWSTroubleshootingTool: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/tools/llm_tools.py
  - [logging_kwargs_std_logger] linha 92: logger.error("Error getting job status: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 142: logger.error("Error getting failed jobs: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 173: logger.error("Error getting job logs: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 223: logger.error("Error getting system health: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 255: logger.error("Error getting job dependencies: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 373: logger.warning("Permission denied: %s", reason, extra=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.
  - [logging_kwargs_std_logger] linha 399: logger.error("Tool execution failed: %s", e, exc_info=
    Correção sugerida: Trocar para `get_logger()` (logger estruturado) OU usar `extra={...}` no logging padrão.

Arquivo: 6.0-new1-local/resync/workflows/workflow_predictive_maintenance.py
  - [await_inside_lock] linha 87: Await inside async-with lock context: _pool_lock
    Correção sugerida: Evitar I/O segurando lock: faça snapshot sob lock e execute awaits fora do lock.

================================================================================
Notas de prioridade (recomendado atacar primeiro)
1) syntax_error e module_package_collision (quebram build/import)
2) await_inside_lock (deadlocks/race em produção)
3) logging_kwargs_std_logger (TypeError em runtime em caminhos de erro)
4) token_in_querystring + hardcoded secrets (segurança)
5) yaml_load_potentially_unsafe / pickle_load / shell=True / SQL f-string (segurança)
6) requests/httpx sem timeout (resiliência)