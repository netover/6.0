# Varredura de duplicações Python — Lote 2 (10 candidatos)

Critério: blocos idênticos de 8 linhas (normalizados), ranqueados por sobreposição entre arquivos.

## 1. `resync/core/health/proactive_monitor.py` ↔ `resync/core/proactive_monitoring.py`
- Blocos compartilhados: **69** | Jaccard: **0.177** | Cobertura: **0.394**
- Risco estimado de consolidação: **médio**
- Ação sugerida: verificar dependências/imports, consolidar função comum em módulo compartilhado, depois remover duplicação residual.

## 2. `resync/core/specialists/tools.py` ↔ `resync/tools/registry.py`
- Blocos compartilhados: **154** | Jaccard: **0.088** | Cobertura: **0.391**
- Risco estimado de consolidação: **médio**
- Ação sugerida: verificar dependências/imports, consolidar função comum em módulo compartilhado, depois remover duplicação residual.

## 3. `resync/core/auto_recovery.py` ↔ `resync/core/health/proactive_monitor.py`
- Blocos compartilhados: **72** | Jaccard: **0.142** | Cobertura: **0.254**
- Risco estimado de consolidação: **médio/alto**
- Ação sugerida: verificar dependências/imports, consolidar função comum em módulo compartilhado, depois remover duplicação residual.

## 4. `resync/core/auto_recovery.py` ↔ `resync/core/proactive_monitoring.py`
- Blocos compartilhados: **36** | Jaccard: **0.083** | Cobertura: **0.206**
- Risco estimado de consolidação: **médio/alto**
- Ação sugerida: verificar dependências/imports, consolidar função comum em módulo compartilhado, depois remover duplicação residual.

## 5. `resync/scripts/update_dashboard_final.py` ↔ `resync/scripts/update_dashboard_refine.py`
- Blocos compartilhados: **12** | Jaccard: **0.078** | Cobertura: **0.176**
- Risco estimado de consolidação: **médio/alto**
- Ação sugerida: verificar dependências/imports, consolidar função comum em módulo compartilhado, depois remover duplicação residual.

## 6. `resync/api/auth_legacy.py` ↔ `resync/api/routes/core/auth.py`
- Blocos compartilhados: **31** | Jaccard: **0.055** | Cobertura: **0.144**
- Risco estimado de consolidação: **médio/alto**
- Ação sugerida: verificar dependências/imports, consolidar função comum em módulo compartilhado, depois remover duplicação residual.

## 7. `resync/core/specialists/tools.py` ↔ `resync/tools/definitions/schemas.py`
- Blocos compartilhados: **21** | Jaccard: **0.013** | Cobertura: **0.137**
- Risco estimado de consolidação: **médio/alto**
- Ação sugerida: verificar dependências/imports, consolidar função comum em módulo compartilhado, depois remover duplicação residual.

## 8. `resync/core/health/health_checkers/filesystem_health_checker.py` ↔ `resync/core/health/health_checkers/memory_health_checker.py`
- Blocos compartilhados: **7** | Jaccard: **0.048** | Cobertura: **0.092**
- Risco estimado de consolidação: **médio/alto**
- Ação sugerida: verificar dependências/imports, consolidar função comum em módulo compartilhado, depois remover duplicação residual.

## 9. `resync/core/health/health_checkers/memory_health_checker.py` ↔ `resync/core/health/health_checkers/websocket_pool_health_checker.py`
- Blocos compartilhados: **5** | Jaccard: **0.038** | Cobertura: **0.083**
- Risco estimado de consolidação: **médio/alto**
- Ação sugerida: verificar dependências/imports, consolidar função comum em módulo compartilhado, depois remover duplicação residual.

## 10. `resync/core/health/health_checkers/filesystem_health_checker.py` ↔ `resync/core/health/health_checkers/websocket_pool_health_checker.py`
- Blocos compartilhados: **5** | Jaccard: **0.038** | Cobertura: **0.083**
- Risco estimado de consolidação: **médio/alto**
- Ação sugerida: verificar dependências/imports, consolidar função comum em módulo compartilhado, depois remover duplicação residual.

