---
name: monitoring-workstations
description: Monitoramento em tempo real de workstations e métricas do TWS. Use para alertas, métricas e acompanhamento.
---

# Monitoring Workstations

## Workflow
1. Coletar métricas (`get_system_metrics`) e status (`get_workstation_status`).
2. Identificar degradação (queda success rate, aumento queue, ws offline).
3. Recomendar ações: priorização, verificação de conectividade, capacidade.
