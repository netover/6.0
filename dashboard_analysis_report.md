# Relatório de Análise: Sistema de Dashboard Resync

## 1. Visão Geral da Arquitetura
O sistema de dashboard do Resync é dividido em duas frentes principais:
*   **Monitoramento de Métricas Internas (Metrics Dashboard):** Focado em performance do sistema (Cache, Router, Reranker, API Latency).
*   **Monitoramento Operacional (Realtime Dashboard):** Focado no TWS (Workstations, Jobs, Orchestration).

Ambos utilizam **WebSockets** para atualizações em tempo real, evitando polling excessivo no frontend.

### Tecnologias Identificadas
*   **Backend:** FastAPI, WebSockets, `asyncio`, `psutil`.
*   **Frontend:** HTML5, CSS3 (Neumorphism Design), Vanilla JavaScript (sem frameworks pesados como React/Vue), Chart.js (inferido pelo contexto de métricas).
*   **Dados:** `MetricSample` e `DashboardMetricsStore` (in-memory com locks).

## 2. Pontos Fortes
*   **Eficiência de Recursos:** O uso de WebSockets reduz a carga no servidor comparado a polling HTTP tradicional.
*   **Design Moderno:** O uso de CSS Neumórfico (`style-neumorphic.css`) oferece uma interface visualmente agradável e moderna.
*   **Independência:** O frontend é leve e não requer build steps complexos (npm/webpack), facilitando manutenção rápida.
*   **Resiliência:** O código JavaScript inclui lógica de reconexão automática (`ws.onclose ... setTimeout(connect, 3000)`).

## 3. Pontos de Atenção e Riscos (Backend)
*   **Redundância de Código:** Existem dois arquivos principais (`dashboard.py` e `metrics_dashboard.py`) com lógicas de WebSocket muito similares mas implementações divergentes.
    *   `metrics_dashboard.py` implementa um `ConnectionManager` class.
    *   `dashboard.py` usa uma lista global `connected_clients` protegida por `asyncio.Lock`.
    *   **Recomendação:** Unificar a gestão de conexões WebSocket em um único serviço reutilizável.
*   **Segurança WebSocket:**
    *   Ambos implementam verificação de autenticação (`_verify_ws_admin`), o que é positivo.
    *   No entanto, a desconexão e limpeza de clientes mortos pode ser aprimorada para evitar memory leaks em conexões "zumbis".
*   **Concorrência:** O uso de locks globais (`_metrics_store._lock`) pode se tornar um gargalo se o número de leituras aumentar muito. Considere usar estruturas imutáveis para snapshots.

## 4. Pontos de Atenção e Riscos (Frontend)
*   **Hardcoded URLs:** O JavaScript constrói a URL do WebSocket (`/api/v1/monitoring/ws`) diretamente. Se a API mudar de prefixo, o frontend quebrará. Recomenda-se injetar essas URLs via template variables.
*   **Escalabilidade de UI:** A lista de logs (`log-panel`) e tabelas crescem indefinidamente no DOM. Em uma sessão longa, isso pode travar o navegador.
    *   **Recomendação:** Implementar "windowing" ou limpar elementos antigos (ex: manter apenas os últimos 100 logs).
*   **Acessibilidade:** O design neumórfico muitas vezes sofre com baixo contraste, dificultando a leitura para usuários com deficiência visual.

## 5. Integração com Novos Recursos
*   **E-mail/Relatórios:** O dashboard atual mostra dados, mas não tem ações.
    *   **Oportunidade:** Adicionar um botão "Exportar Relatório" que chama o endpoint `/admin/notifications/smtp/test` ou (melhor) um novo endpoint que gera o relatório com os dados atuais da tela e envia por e-mail usando o `EmailService`.
*   **Capacity Forecasting:** Os dados de previsão de capacidade gerados pelo workflow não parecem ter uma visualização dedicada no dashboard atual. Seria um excelente widget para adicionar à `jobs-panel`.

## 6. Plano de Ação Recomendado
1.  **Refatoração Backend:** Unificar `metrics_dashboard.py` e `dashboard.py` para usar um `WebSocketManager` comum em `resync/core/websocket.py`.
2.  **Melhoria Frontend:** Adicionar limite de itens no DOM para logs e tabelas.
3.  **Nova Feature:** Criar widget "Capacity Forecast" consumindo o resultado do workflow.
4.  **Ação de E-mail:** Adicionar botão no header para "Email Report".
