# 🔬 Análise do todo.md — alinhamento com agent_manager.py

Este documento foi revisado para refletir o estado real do desenvolvimento do módulo `resync/core/agent_manager.py` e corrigir inconsistências existentes.

***

## ✅ Escopo Atual

- O conteúdo anterior deste arquivo tratava exclusivamente de hardening do endpoint CSP em `app_factory.py`.
- Não havia qualquer item relacionado ao `agent_manager.py`, portanto o todo estava **desalinhado do módulo solicitado**.
- Abaixo está a validação real do `agent_manager.py` e as pendências alinhadas ao código atual.

***

## ✅ Status CSP/app_factory (referência histórica)

- **Exception poisoning (P0)**: resolvido (_handle_csp_report removido; parsing type-safe aplicado).
- **Type guard no endpoint (P1)**: resolvido (validação explícita antes de processar).
- **Status:** concluído e consistente com o código atual.

***

## ✅ Validação de Afirmações do agent_manager.py

- **“Removed singleton anti-pattern”**: **parcialmente verdadeiro**. Há cache global `_agent_manager` com fallback em `get_agent_manager()` e lock de inicialização. Não é singleton rígido, mas ainda existe estado global.
- **“Per-session history”**: **verdadeiro**. `UnifiedAgent` mantém histórico por `conversation_id` em `_histories`.
- **“Tools filtered by config”**: **verdadeiro**. `_tools_for_config` filtra as ferramentas por lista permitida no `AgentConfig`.
- **“Structured logging throughout”**: **verdadeiro**. Logs estruturados em inicialização, criação de agentes e roteamento.

***

## ✅ Pendências Implementadas (agent_manager.py)

### P1 — Alto

1. **Inicialização do TWS client não bloqueante**  
   - Aplicado: `tws_client_factory` executado via `asyncio.to_thread` ou `await` se async.

2. **Cancelamento de tasks**  
   - Aplicado: `asyncio.CancelledError` é repropagado em `Agent.arun`.

3. **Uso de atributo privado `_loop` em locks**  
   - Aplicado: locks são cacheados por loop sem acesso a atributo privado.

### P2 — Médio

4. **YAML parsing em thread**  
   - Aplicado: `yaml.safe_load` executa via `asyncio.to_thread`.

5. **Documentação do padrão de instância**  
   - Aplicado: docstring atualizada para “cached module-level instance”.

***

## ✅ Próximas Etapas Recomendadas

- Nenhuma pendência técnica restante neste módulo.
