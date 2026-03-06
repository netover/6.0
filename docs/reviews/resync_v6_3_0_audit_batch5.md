# Revisão técnica — Resync v6.3.0 (lote 5: próximos 100 arquivos não-API)

## Escopo auditado
- Rodada focada em 100 arquivos fora de `resync/api/*`, com verificação automatizada para erros de runtime, segurança e robustez.
- Comando-base da auditoria: `PYENV_VERSION=3.14.0 python -m ruff check --select F821,F823,B904,B905,B006,B007,B008,S,PLE $(cat /tmp/next100.txt)`.

## Findings confirmados e status

_Status da auditoria: 2026-03-06 (itens abaixo corrigidos nesta rodada)._ 

1) **[🔴 Crítico] Tipos não resolvidos em `background_tasks.py` (`threading`/`ConversationMemory`)** — **Corrigido**.
- Risco: `F821` e potencial falha em análise estática estrita, reduzindo confiabilidade do pipeline de release.
- Correção: import explícito de `threading` e `TYPE_CHECKING` com import condicional de `ConversationMemory`.

2) **[🟡 Resiliência] `zip()` sem `strict` em `audit_lock.py`** — **Corrigido**.
- Risco: desalinhamento silencioso entre chaves e TTLs em cleanup de lock.
- Correção: `zip(decoded_keys, ttls, strict=False)` explícito.

3) **[⚪ Qualidade/Observabilidade] `except ...: pass` em serialização de tool call (`agent_manager.py`)** — **Corrigido**.
- Risco: perda de diagnóstico em falhas de serialização de objeto retornado por LLM/tooling.
- Correção: logs `debug` com `exc_info=True` nos dois caminhos de fallback (`model_dump` e `dict`).

4) **[⚪ Qualidade/Observabilidade] `except ...: pass` no fallback de `hexpire` (`valkey_client.py`)** — **Corrigido**.
- Risco: incapacidade de rastrear incompatibilidade de cliente Valkey no fallback de HFE.
- Correção: log `debug` estruturado (`valkey_hexpire_not_supported`) com contexto e stack trace.

5) **[⚪ Qualidade] Variável de loop não utilizada em `app_factory.py`** — **Corrigido**.
- Risco: ruído de lint e manutenção.
- Correção: renomeada para `_log_name` em ambos os loops de registro de rotas.

6) **[⚪ Qualidade] `S311` em módulo de chaos engineering** — **Classificado como uso intencional**.
- Contexto: uso de `random` para simulação/fuzzing, não para propósito criptográfico.
- Tratativa: adicionado `# noqa: S311` com justificativa inline nos pontos sinalizados.

## Observações
- A rodada seguiu a diretriz de não ignorar alertas “fora do escopo” quando havia evidência prática no código.
- Após correções, o conjunto de checks acima ficou sem pendências.
