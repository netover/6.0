---
name: managing-jobs
description: Executa/para/reexecuta jobs e orienta procedimentos seguros no TWS. Use para run/stop/rerun/submit/release/hold.
---

# Managing Jobs

## Regras de segurança
- Para ações destrutivas (stop/kill/cancel), exigir confirmação humana (HITL).
- Sempre confirmar o job_name e janela/ambiente antes de executar.

## Passos
1. Validar estado do job.
2. Validar impacto (dependências, downstream).
3. Executar ação apropriada e verificar resultado.
