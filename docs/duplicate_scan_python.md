# Varredura de duplicações em arquivos Python

Total de arquivos analisados: **122**

## Duplicações API (top-level) x API Routes

- `resync/api/health.py` ↔ `resync/api/routes/core/health.py`: **0.905**

## Candidatos seguros para ajuste/remoção

- Risco **alto**: consolidar `resync/api/health.py` em `resync/api/routes/core/health.py` (sim 0.905) após validar imports.
