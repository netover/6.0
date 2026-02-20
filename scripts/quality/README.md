# Quality Gates (Fases 1–4)

Este diretório implementa a execução incremental do plano de qualidade:

## Fase 1 (mypy strict progressivo)
- Baseline por lote: `config/quality/mypy_baseline.json`
- Gate de regressão: `python scripts/quality/check_mypy_regression.py`
- Opcional para exigir redução em PR: `MYPY_REQUIRE_REDUCTION=1`

## Fase 2 (cobertura em rampa)
- Configuração da rampa: `config/quality/coverage_ramp.json`
- Execute:
  1. `PYTHONPATH=. pytest -q --cov=resync --cov-report=xml --cov-report=term`
  2. `python scripts/quality/check_coverage_ramp.py`

## Fase 3 (segurança)
- Semgrep:
  - gerar relatório: `semgrep --config auto --json --output semgrep-report.json resync`
  - gate de novos achados: `python scripts/quality/check_semgrep_regression.py`
  - atualizar baseline (controlado): `python scripts/quality/update_semgrep_baseline.py`
- Bandit (high/critical): `bandit -r resync -q -lll`
- pip-audit: `pip-audit` e `pip-audit -r requirements.txt`

## Fase 4 (higiene contínua)
- `deptry . --json-output deptry-report.json`
- `vulture resync --min-confidence 90 --sort-by-size`
- `radon cc -s -a resync`
- `radon mi -s resync`
