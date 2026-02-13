---
name: analyzing-job-logs
description: Analisa logs de jobs do TWS, identifica códigos de erro (RC, ABEND) e sugere correções. Use quando o usuário perguntar por que um job falhou, relatar um ABEND ou pedir análise de log.
---

# Analyzing Job Logs

## Quick Start
Para analisar um job que falhou:
1. Extraia o log usando a ferramenta `get_job_log`.
2. Identifique o código de erro (Ex: RC=8 ou ABEND S0C4).
3. Busque a causa raiz usando a ferramenta `analyze_return_code` ou `lookup_error`.

## Workflow de Troubleshooting (Feedback Loop)
1. **Validar:** Verifique o status do job. Se estiver SUCC, não há erro.
2. **Analisar:** Se ABEND, isole a mensagem `+STEP` que falhou.
3. **Resolver:** Siga as práticas padrão de resolução do TWS para o código encontrado.
