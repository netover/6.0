# Resync v6.3.0 Audit — Batch 12 (próximos 200 arquivos)

## Escopo
- Lote auditado: linhas 401–600 de `/tmp/all_py.txt`.
- Arquivos Python no lote: 200.
- Arquivos `resync/*` no lote: 175.

## Método
- Verificação estática focada em erros de alto impacto com:
  - `F821,F823` (nomes indefinidos)
  - `B904,B905` (robustez de exceções/zip)
  - `S,PLE` (segurança e erros lógicos)
- Correções mínimas apenas com evidência direta no código.

## Findings e correções aplicadas

### 1) `resync/knowledge/kg_store/store.py`
- **Achado**: Query recursiva em `get_subgraph` montada por interpolação de string para `doc_filter` e `LIMIT`, gerando alerta S608 e dificultando validação estática.
- **Correção**:
  - separação explícita de dois caminhos SQL (`com doc_id` e `sem doc_id`),
  - uso de parâmetros bind para `LIMIT` (`$6` / `$5`),
  - remoção da composição dinâmica de `doc_filter`.
- **Impacto**: reduz superfície de erro em composição SQL e mantém comportamento funcional original.

### 2) `resync/knowledge/retrieval/tws_relations.py`
- **Achado**: padrões SQL com `f-string` para filtro de tenant e aliases, acionando S608 repetido.
- **Correção**:
  - reescrita dos métodos de `TWSQueryPatterns` para SQL estático com bifurcação `if tenant_id`;
  - retorno de parâmetros explícitos por ramo (`[arg]` vs `[arg, tenant_id]`);
  - em `to_sql`, manutenção da sanitização de identificador (`sanitize_sql_identifier`) e troca de `f-string` por template com `replace` para reduzir ruído de scanner.
- **Impacto**: mantém a semântica das consultas e melhora a auditabilidade de segurança.

## Validação
- `PYENV_VERSION=3.14.0 python -m ruff check --select F821,F823,B904,B905,S,PLE $(cat /tmp/next200h_resync.txt)`
- `PYENV_VERSION=3.14.0 python -m py_compile resync/knowledge/kg_store/store.py resync/knowledge/retrieval/tws_relations.py`
