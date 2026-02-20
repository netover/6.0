# Validação conforme temp1.md

## TASK-000: Bootstrap da Revisão
- [x] Gerar mapa completo do projeto (files_to_review.txt)
- [x] Contar total de arquivos (wc -l files_to_review.txt)
- [x] Gerar inventário de dependências (deps_snapshot.txt)
- [x] Executar safety check (requirements.txt)
- [x] Executar bandit e gerar bandit_report.json
- [x] Gerar snapshot de complexidade (complexity_report.txt)
- [ ] Gerar mapa de imports circulares (deps_graph.svg)
- [x] Criar tracking sheet de arquivos

Total de arquivos .py: 537

Tracking sheet:

| # | Arquivo | Severidade | Status | Observações |
|---|---|---|---|---|
| 001 | analyze_pr_comments.py | Média | [x] | Exceções genéricas removidas; validações estruturais e fallback de campos |
| 002-101 | batch_api_core_inicial (100 arquivos) | Baixa/Média | [x] | Compilação sintática ok (100/100); triagem detectou 50 usos de except Exception para correção em ondas |
| 102-201 | batch_api_core_segundo (100 arquivos) | Baixa/Média | [x] | Compilação sintática ok (100/100); correções aplicadas em segurança/middleware/health para preservar HTTPException e reduzir except genérico |

## STEP 1 — Leitura Estrutural
- [ ] Ler cabeçalho e docstring do módulo
- [ ] Mapear imports (usados/não usados, relativos/absolutos, condicionais, circulares)
- [ ] Listar classes/funções/constantes
- [ ] Verificar coesão do módulo (SRP, tamanho, naming)

## STEP 2 — Revisão Linha a Linha
- [ ] Constantes e configurações no topo
- [ ] Revisar cada classe (init, herança, atributos, métodos, dataclass/Pydantic)
- [ ] Revisar cada função/método (assinatura, docstring, tamanho, complexidade, parâmetros, retorno, lógica)

## STEP 3 — Auditoria Async/Concorrência
- [ ] Identificar contexto de execução
- [ ] Validar I/O com await e ausência de blocking calls
- [ ] Revisar task management (create_task, gather, timeout, TaskGroup)
- [ ] Revisar shared state (locks/semaphores, race conditions)
- [ ] Revisar context vars

## STEP 4 — Auditoria de Segurança
- [ ] Injection risks (SQL/NoSQL/command/SSTI/path traversal)
- [ ] Autenticação e autorização (Depends, permissões, JWT, tokens em logs)
- [ ] Dados sensíveis (secrets, PII, stack trace, repr/str)
- [ ] Rate limiting e DoS (limites, payload, throttling)
- [ ] Dependências (versions pinadas, CVEs)

## STEP 5 — Auditoria de Performance
- [ ] Database/Cache (N+1, SELECT *, índices, cache)
- [ ] Serialização/Deserialização (pydantic, json em loops, cópias)
- [ ] Memory (listas enormes, closures, leaks, globais)
- [ ] I/O patterns (gather, retry/backoff, pool)

## STEP 6 — Auditoria de Resiliência
- [ ] Error handling (bare except, re-raise, tipos)
- [ ] Timeouts (HTTP/DB/Redis/externos)
- [ ] Retry logic (backoff, jitter, max retries, idempotência)
- [ ] Circuit breaker (fallback, métricas)

## STEP 7 — Auditoria de Observabilidade
- [ ] Logging (estruturado, nível, request_id, mascaramento)
- [ ] Métricas (contadores, histogramas, gauges)
- [ ] Tracing (spans, atributos, propagação)
- [ ] Health signals (alertas, degradação)

## STEP 8 — Auditoria de Testes
- [ ] Cobertura do arquivo
- [ ] Qualidade dos testes (behavior, paths, edge cases)
- [ ] Ausências críticas (sem testes unitários ou concorrência)

## STEP 9 — Output da Task (Relatório Padrão)
- [ ] Preencher relatório padrão por arquivo

## Checklist Final de Review Completa
- [ ] Todos os TASK-NNN concluídos e relatórios gerados
- [ ] Todos os problemas críticos corrigidos e re-revisados
- [ ] Todos os problemas altos com issue e owner
- [ ] bandit_report.json sem severidade HIGH/MEDIUM
- [ ] Cobertura de testes ≥ 80% nos módulos críticos
- [ ] Nenhum secret hardcoded (grep -r "password|secret|token" .)
- [ ] requirements.txt com versões pinadas
- [ ] ADRs/documentação arquitetural relevantes
- [ ] CHANGELOG.md atualizado
