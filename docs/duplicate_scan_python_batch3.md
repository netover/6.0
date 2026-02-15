# Varredura de duplicações Python — Lote 3 (10 candidatos)

Critério: blocos idênticos de 8 linhas (normalizados), excluindo pares já listados no Lote 2.

## 1. `resync/core/health/health_checkers/cpu_health_checker.py` ↔ `resync/core/health/health_checkers/memory_health_checker.py`
- Blocos compartilhados: **6** | Jaccard: **0.039** | Cobertura: **0.079** | Score: **0.059**
- Risco estimado de consolidação: **alto**
- Ação sugerida: não remover arquivos; tratar apenas duplicações pontuais de baixo risco.

## 2. `resync/core/health/health_checkers/cpu_health_checker.py` ↔ `resync/core/health/health_checkers/filesystem_health_checker.py`
- Blocos compartilhados: **6** | Jaccard: **0.039** | Cobertura: **0.079** | Score: **0.059**
- Risco estimado de consolidação: **alto**
- Ação sugerida: não remover arquivos; tratar apenas duplicações pontuais de baixo risco.

## 3. `resync/core/health/health_checkers/redis_health_checker.py` ↔ `resync/core/health/health_checkers/websocket_pool_health_checker.py`
- Blocos compartilhados: **5** | Jaccard: **0.033** | Cobertura: **0.083** | Score: **0.058**
- Risco estimado de consolidação: **alto**
- Ação sugerida: não remover arquivos; tratar apenas duplicações pontuais de baixo risco.

## 4. `resync/core/health/health_checkers/connection_pools_health_checker.py` ↔ `resync/core/health/health_checkers/websocket_pool_health_checker.py`
- Blocos compartilhados: **5** | Jaccard: **0.031** | Cobertura: **0.083** | Score: **0.057**
- Risco estimado de consolidação: **alto**
- Ação sugerida: não remover arquivos; tratar apenas duplicações pontuais de baixo risco.

## 5. `resync/core/health/health_checkers/health_checker_factory.py` ↔ `resync/core/health/health_config_manager.py`
- Blocos compartilhados: **12** | Jaccard: **0.025** | Cobertura: **0.088** | Score: **0.056**
- Risco estimado de consolidação: **alto**
- Ação sugerida: não remover arquivos; tratar apenas duplicações pontuais de baixo risco.

## 6. `resync/core/health/health_checkers/connection_pools_health_checker.py` ↔ `resync/core/health/health_checkers/database_health_checker.py`
- Blocos compartilhados: **8** | Jaccard: **0.035** | Cobertura: **0.075** | Score: **0.055**
- Risco estimado de consolidação: **alto**
- Ação sugerida: não remover arquivos; tratar apenas duplicações pontuais de baixo risco.

## 7. `resync/core/health/health_checkers/database_health_checker.py` ↔ `resync/core/health/health_checkers/websocket_pool_health_checker.py`
- Blocos compartilhados: **5** | Jaccard: **0.027** | Cobertura: **0.083** | Score: **0.055**
- Risco estimado de consolidação: **alto**
- Ação sugerida: não remover arquivos; tratar apenas duplicações pontuais de baixo risco.

## 8. `resync/core/health/health_checkers/memory_health_checker.py` ↔ `resync/core/health/health_checkers/redis_health_checker.py`
- Blocos compartilhados: **5** | Jaccard: **0.030** | Cobertura: **0.066** | Score: **0.048**
- Risco estimado de consolidação: **alto**
- Ação sugerida: não remover arquivos; tratar apenas duplicações pontuais de baixo risco.

## 9. `resync/core/health/health_checkers/filesystem_health_checker.py` ↔ `resync/core/health/health_checkers/redis_health_checker.py`
- Blocos compartilhados: **5** | Jaccard: **0.030** | Cobertura: **0.066** | Score: **0.048**
- Risco estimado de consolidação: **alto**
- Ação sugerida: não remover arquivos; tratar apenas duplicações pontuais de baixo risco.

## 10. `resync/core/health/health_checkers/cpu_health_checker.py` ↔ `resync/core/health/health_checkers/websocket_pool_health_checker.py`
- Blocos compartilhados: **4** | Jaccard: **0.029** | Cobertura: **0.067** | Score: **0.048**
- Risco estimado de consolidação: **alto**
- Ação sugerida: não remover arquivos; tratar apenas duplicações pontuais de baixo risco.

## Ajustes pontuais aplicados nesta rodada

- Extraído utilitário comum `resync/core/health/health_checkers/common.py` para reduzir duplicação de:
  - cálculo de tempo de resposta (`response_time_ms`);
  - classificação por thresholds (`threshold_status`);
  - construção padronizada de erro (`build_error_health`).
- Aplicado nos checkers:
  - `cpu_health_checker.py`
  - `memory_health_checker.py`
  - `filesystem_health_checker.py`
  - `websocket_pool_health_checker.py`
- Efeito colateral positivo: mensagens de threshold agora são formatadas corretamente com o valor medido.
