# Plano de Correção - settings.py

## Problemas Validados e Correções Planejadas

### 1. Credenciais Hardcoded (CRÍTICO - Prioridade Alta)
**Status:** CONFIRMADO
**Arquivo:** `resync/settings.py`

**Correção Planejada:**
- [ ] Remover default `secret_key = SecretStr("CHANGE_ME_IN_PRODUCTION_USE_ENV_VAR")` (linha 739)
- [ ] Alterar para `default=None` (obrigatório via env var)
- [ ] Remover default `admin_password = None` - manter como optional mas com validação mais robusta
- [ ] Manter validadores existentes (já implementados em settings_validators.py)

**Nota:** Os validadores JA existem e funcionam. Precisamos apenas remover os defaults inseguros.

---

### 2. Dependências Circulares Potenciais (Parcial - Prioridade Média)
**Status:** PARCIAL - `_LAZY_IMPORTS` está vazio, risco futuro
**Arquivo:** `resync/settings.py`

**Correção Planejada:**
- [ ] Remover código PEP 562 `__getattr__` (linhas 1601-1614) já que não está em uso
- [ ] Remover `_LAZY_IMPORTS` e `_LOADED_IMPORTS` (linhas 1596-1598)
- [ ] Limpar imports não utilizados (`importlib`, `threading`)

---

### 3. Pool Sizing para Single VM (Moderado - Prioridade Média)
**Status:** CONFIRMADO - valores podem ser altos para recursos limitados
**Arquivo:** `resync/settings.py`

**Correção Planejada:**
- [ ] Reduzir `db_pool_min_size` de 5 para 2
- [ ] Reduzir `db_pool_max_size` de 20 para 10
- [ ] Reduzir `redis_pool_min_size` de 2 para 1
- [ ] Reduzir `redis_pool_max_size` de 10 para 5

---

### 4. Campos Deprecated (Moderado - Prioridade Baixa)
**Status:** CONFIRMADO
**Arquivo:** `resync/settings.py`

**Correção Planejada:**
- [ ] Remover completamente `context_db_path` (linhas 123-127)
- [ ] Atualizar documento de CHANGELOG/MIGRATION

---

### 5. Regex de Versão Excessivamente Complexo (Moderado - Prioridade Baixa)
**Status:** CONFIRMADO
**Arquivo:** `resync/settings.py`

**Correção Planejada:**
- [ ] Simplificar pattern de semver para apenas `r"^\d+\.\d+\.\d+$"` (X.Y.Z)
- [ ] Atualizar description para refletir que não suporta pre-release

---

### 6. Feature Flags sem Documentação Clara (Moderado - Prioridade Baixa)
**Status:** CONFIRMADO
**Arquivo:** `resync/settings.py`

**Correção Planejada:**
- [ ] Adicionar data de deprecação nas descriptions das 4 migration flags:
  - MIGRATION_USE_NEW_CACHE
  - MIGRATION_USE_NEW_TWS
  - MIGRATION_USE_NEW_RATE_LIMIT
  - MIGRATION_ENABLE_METRICS
- [ ] Adicionar target de remoção (ex: "Remover após Q2 2026")

---

## Ordem de Execução Sugerida

| Fase | Problema | Tempo Est. |
|------|----------|------------|
| 1 | Credenciais Hardcoded | 15 min |
| 2 | Dependências Circulares | 10 min |
| 3 | Pool Sizing | 10 min |
| 4 | Campos Deprecated | 5 min |
| 5 | Regex Versão | 5 min |
| 6 | Feature Flags | 10 min |

**Total estimado:** ~55 minutos

---

## NÃO FAZER (Já Implementado)
- Validação de produção para secrets (JA EXISTE em settings_validators.py)
- Model validators cross-field (JA EXISTE)
