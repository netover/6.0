document.addEventListener('DOMContentLoaded', () => {
    const reviewList = document.getElementById('review-list');
    const statusMessage = document.getElementById('status-message');
    const filterStatus = document.getElementById('filter-status');
    const searchQuery = document.getElementById('search-query');
    const applyFiltersBtn = document.getElementById('apply-filters');

    const metricPending = document.getElementById('metric-pending');
    const metricApproved = document.getElementById('metric-approved');
    const metricRejected = document.getElementById('metric-rejected');

    async function loadAudits(status = 'pending', query = '') {
        reviewList.innerHTML = '';
        statusMessage.textContent = 'Carregando revisões...';
        try {
            let url = `/api/audit/flags?status=${status}`;
            if (query) {
                url += `&query=${encodeURIComponent(query)}`;
            }

            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const items = await response.json();

            if (items.length === 0) {
                statusMessage.textContent = 'Nenhuma revisão encontrada com os filtros aplicados.';
                return;
            }

            items.forEach((item) => {
                const reviewItem = document.createElement('div');
                reviewItem.className = 'review-item';
                reviewItem.id = `item-${item.memory_id}`;

                const memoryId = document.createElement('p');
                memoryId.innerHTML = `<span class="label">ID da Memória:</span> ${item.memory_id}`;
                reviewItem.appendChild(memoryId);

                const status = document.createElement('p');
                status.innerHTML = `<span class="label">Status:</span> ${item.status}`;
                reviewItem.appendChild(status);

                const userQueryLabel = document.createElement('p');
                userQueryLabel.innerHTML = `<span class="label">Pergunta do Usuário:</span>`;
                reviewItem.appendChild(userQueryLabel);

                const userQuery = document.createElement('pre');
                userQuery.textContent = item.user_query;
                reviewItem.appendChild(userQuery);

                const agentResponseLabel = document.createElement('p');
                agentResponseLabel.innerHTML = `<span class="label">Resposta do Agente:</span>`;
                reviewItem.appendChild(agentResponseLabel);

                const agentResponse = document.createElement('pre');
                agentResponse.textContent = item.agent_response;
                reviewItem.appendChild(agentResponse);

                const reason = document.createElement('p');
                reason.innerHTML = `<span class="label">Motivo da Sinalização (IA):</span> ${escapeHtml(item.ia_audit_reason || 'N/A')} (Confiança: ${item.ia_audit_confidence || 'N/A'})`;
                reviewItem.appendChild(reason);

                const createdAt = document.createElement('p');
                createdAt.innerHTML = `<span class="label">Criado em:</span> ${new Date(item.created_at).toLocaleString()}`;
                reviewItem.appendChild(createdAt);

                if (item.reviewed_at) {
                    const reviewedAt = document.createElement('p');
                    reviewedAt.innerHTML = `<span class="label">Revisado em:</span> ${new Date(item.reviewed_at).toLocaleString()}`;
                    reviewItem.appendChild(reviewedAt);
                }

                const actions = document.createElement('div');
                actions.className = 'actions';
                if (item.status === 'pending') {
                    const approveBtn = document.createElement('button');
                    approveBtn.className = 'approve-btn';
                    approveBtn.dataset.id = item.memory_id;
                    approveBtn.textContent = 'Aprovar';
                    actions.appendChild(approveBtn);

                    const rejectBtn = document.createElement('button');
                    rejectBtn.className = 'reject-btn';
                    rejectBtn.dataset.id = item.memory_id;
                    rejectBtn.textContent = 'Rejeitar';
                    actions.appendChild(rejectBtn);
                }
                reviewItem.appendChild(actions);

                const statusDiv = document.createElement('div');
                statusDiv.className = 'status';
                reviewItem.appendChild(statusDiv);

                reviewList.appendChild(reviewItem);
            });
            statusMessage.textContent = '';

        } catch (error) {
            statusMessage.textContent = 'Erro ao carregar revisões.';
            console.error(error);
        }
    }

    async function loadMetrics() {
        try {
            const response = await fetch('/api/audit/metrics'); // New endpoint for metrics
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const metrics = await response.json();
            metricPending.textContent = metrics.pending;
            metricApproved.textContent = metrics.approved;
            metricRejected.textContent = metrics.rejected;
        } catch (error) {
            console.error('Erro ao carregar métricas:', error);
            metricPending.textContent = 'N/A';
            metricApproved.textContent = 'N/A';
            metricRejected.textContent = 'N/A';
        }
    }

    applyFiltersBtn.addEventListener('click', () => {
        const status = filterStatus.value;
        const query = searchQuery.value;
        loadAudits(status, query);
    });

    reviewList.addEventListener('click', async (event) => {
        if (event.target.matches('.approve-btn') || event.target.matches('.reject-btn')) {
            const memoryId = event.target.dataset.id;
            const action = event.target.matches('.approve-btn') ? 'approve' : 'reject';
            const itemDiv = document.getElementById(`item-${memoryId}`);
            const statusDiv = itemDiv.querySelector('.status');

            // Disable buttons
            itemDiv.querySelector('.actions').innerHTML = '';
            statusDiv.textContent = 'Processando...';

            try {
                const response = await fetch('/api/audit/review', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        memory_id: memoryId,
                        action: action,
                    }),
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Falha na requisição');
                }

                const result = await response.json();
                statusDiv.textContent = `Ação '${result.status}' concluída com sucesso.`;
                // Reload audits and metrics after successful review
                loadAudits(filterStatus.value, searchQuery.value);
                loadMetrics();

            } catch (error) {
                statusDiv.textContent = `Erro: ${error.message}`;
                console.error(error);
            }
        }
    });

    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') return '';
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
     }

    // Initial load
    loadAudits();
    loadMetrics();
});