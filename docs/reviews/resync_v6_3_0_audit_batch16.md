# Resync v6.3.0 Audit — Batch 16 (ajustes pós-comentários de PR)

## Escopo
- Revisão dos pontos levantados após o patch de hardening WebSocket.
- Ajustes mínimos para alinhamento com segurança/ASGI sem alterar lógica de negócio.

## Correções aplicadas

### 1) `session_id` sem exposição de `user_id` bruto no chat WS
- O helper `_session_id_for_websocket()` usava `user_id` em texto puro ao compor `session_id`.
- Alterado para `hash_user_id(...)` antes de compor o identificador de sessão.
- Impacto: reduz exposição de identificador sensível em payloads/logs/chaves de contexto.

### 2) Rejeição WS com sequência ASGI consistente no monitoring dashboard
- Nos caminhos de rejeição por auth e por limite de conexões, o endpoint chamava `close()` antes de `accept()`.
- Ajustado para `accept()` seguido de `close(...)`, mantendo consistência com os demais hardenings de WS já aplicados no projeto.
- Impacto: evita risco de `RuntimeError` em implementações ASGI mais estritas.

## Validação executada
- `PYENV_VERSION=3.14.0 python -m ruff check --select F821,F823,B904,B905,S,PLE resync/api/chat.py resync/api/routes/monitoring/dashboard.py`
- `PYENV_VERSION=3.14.0 python -m py_compile resync/api/chat.py resync/api/routes/monitoring/dashboard.py`
