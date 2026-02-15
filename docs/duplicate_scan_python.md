# Varredura de duplicações em arquivos Python

Total de arquivos analisados: **122**

## Duplicações API (top-level) x API Routes

- `resync/api/auth_legacy.py` ↔ `resync/api/routes/core/auth.py`: **0.464** (sobreposição parcial)

## Candidatos seguros para ajuste/remoção

- `resync/api/health.py` ↔ `resync/api/routes/core/health.py` foi **consolidado**: imports migrados para `resync/api/routes/core/health.py` e arquivo legado removido.
- Próximo candidato de revisão: consolidar trechos comuns entre `auth_legacy` e `routes/core/auth` com compatibilidade retroativa.
