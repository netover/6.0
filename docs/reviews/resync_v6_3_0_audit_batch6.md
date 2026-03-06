# Revisão técnica — Resync v6.3.0 (lote 6: próximos 100 arquivos + Pydantic v2)

## Escopo auditado
- Continuação da varredura dos próximos 100 arquivos (rodada posterior ao lote 5).
- Foco adicional: validação de compatibilidade Pydantic v2 e remoção de fallback explícito para Pydantic v1.

## Findings confirmados e status

_Status da auditoria: 2026-03-06 (itens corrigidos nesta rodada)._ 

1) **[🔴 Crítico] Símbolos indefinidos no fallback de LLM (`llm_fallback.py`)** — **Corrigido**.
- Evidência: uso de `LLMRateLimitError` sem import e `error_str` sem definição.
- Impacto: `NameError` em runtime durante tratamento de erro de provedor LLM.
- Correção: import de `LLMRateLimitError` e criação explícita de `error_str` antes das comparações.

2) **[🟡 Resiliência/Correção] Mapeamento incorreto de erro de rede para rate-limit (`llm_fallback.py`)** — **Corrigido**.
- Evidência: bloco `except LLMNetworkError` duplicava atualização de métrica/motivo para `RATE_LIMIT`.
- Impacto: classificação incorreta de fallback e telemetria inconsistente.
- Correção: mantida classificação `NETWORK`, contagem de falha por modelo e log dedicado `llm_network_falling_back`.

3) **[🟠 Segurança/Qualidade] Alerta `S311` em jitter de retry (`tws_service.py`)** — **Tratado**.
- Contexto: `random.uniform` usado para full-jitter de backoff (não criptográfico).
- Correção: anotação `# noqa: S311` com justificativa inline.

4) **[⚪ Qualidade] `B904` em teste de cancelamento (`test_task_tracker.py`)** — **Corrigido**.
- Correção: `raise RuntimeError(...) from e` para manter encadeamento de exceção.

5) **[🟢 Práticas/Pydantic v2] Remoção de fallback explícito de Pydantic v1 (`agent_manager.py`)** — **Corrigido**.
- Evidência: bloco de serialização condicional com `tool_call.dict()` marcado como "Pydantic v1".
- Correção: mantido apenas caminho `model_dump()` (Pydantic v2) e fallback estrutural por atributos.

6) **[🟢 Práticas] Ajustes adicionais de robustez nos workflows do lote** — **Corrigido**.
- `zip(..., strict=False)` explícito em cálculos estatísticos.
- `datetime.now(datetime.UTC)` em pontos sinalizados por compatibilidade moderna.
- `raise ... from e` em validações com `except` que reempacotavam erro.

## Verificação Pydantic v2 (estado atual)
- Requisitos do projeto referenciam `pydantic==2.12.0` e `pydantic-settings>=2.10.1,<3.0.0`.
- Busca textual por padrões explícitos de API v1 (`pydantic.v1`, `@validator`, `@root_validator`, `parse_obj`, fallback `.dict()` de Pydantic) não encontrou ocorrências ativas após as correções desta rodada.

## Observações
- Há alertas de segurança em scripts de instalação e testes que refletem execução de subprocesso/segredos de fixture em contexto controlado; eles exigem política específica de hardening separada para serem eliminados sem alterar fluxo operacional desses utilitários.
