# Resync v6.3.0 Audit — Batch 15 (aplicação de correções WebSocket críticas)

## Escopo
- Aplicação das correções propostas na revisão técnica WebSocket (segurança, concorrência, resiliência e observabilidade).
- Arquivos alterados:
  - `resync/api/routes/orchestration.py`
  - `resync/api/websocket/handlers.py`
  - `resync/api/chat.py`
  - `resync/api/routes/monitoring/dashboard.py`
  - `resync/core/websocket_pool_manager.py`

## Correções aplicadas

### 1) Autenticação fail-closed no WS de orquestração
- Adicionada resolução de `user_id` via JWT (`Authorization`/`token` query) antes de iniciar execução.
- Conexão inválida é rejeitada com `WS_1008_POLICY_VIOLATION`.
- Execução agora usa `user_id` autenticado em vez de valor fixo.

### 2) Backpressure para stream de eventos de orquestração
- Fila de eventos alterada para bounded queue (`ORCHESTRATION_WS_QUEUE_MAXSIZE`, default 1000).
- Em overflow (`QueueFull`), evento é descartado com `warning` estruturado para evitar crescimento de memória sem limite.

### 3) Isolamento de sessão em WebSocket de chat
- `session_id` deixou de depender de query param arbitrário do cliente.
- Session id agora é server-side e estável por conexão (`websocket.state.session_id`) com vínculo ao `user_id` quando disponível.

### 4) Hardening de erro e timeout no handler WS de agentes
- `send_personal_message` agora aplica timeout com `asyncio.wait_for`.
- Falhas de LLM não retornam `str(exception)` ao cliente; resposta agora é genérica e detalhe fica em `logger.exception`.

### 5) CancelledError e higiene de datetime
- Corrigido bloco `except asyncio.CancelledError` com log executável (sem código morto após `raise`) no dashboard de monitoring.
- Normalização de timestamps para `datetime.UTC` no `websocket_pool_manager`.

## Validação executada
- `PYENV_VERSION=3.14.0 python -m ruff check --select F821,F823,B904,B905,S,PLE resync/api/routes/orchestration.py resync/api/websocket/handlers.py resync/api/chat.py resync/api/routes/monitoring/dashboard.py resync/core/websocket_pool_manager.py`
- `PYENV_VERSION=3.14.0 python -m py_compile resync/api/routes/orchestration.py resync/api/websocket/handlers.py resync/api/chat.py resync/api/routes/monitoring/dashboard.py resync/core/websocket_pool_manager.py`
- `PYENV_VERSION=3.14.0 pytest -q resync/tests/test_jwt_utils.py` *(falhou por limitação de ambiente: plugin `pytest_asyncio` ausente)*
