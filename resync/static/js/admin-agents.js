/**
 * Admin Agents Module
 * Provides views for agent inventory and validation status.
 */
class AdminAgents {
  constructor(app) {
    this.app = app;
    this.api = app.api;
  }

  async loadRegistryView() {
    const content = document.getElementById('content');
    content.innerHTML = '<div class="card"><div class="stat-label">Carregando Agent Registry...</div></div>';

    try {
      const data = await this.api.get('/admin/agents/registry');
      const summary = data.summary || {};
      const entries = data.entries || [];

      const missingPrompts = (summary.missing_prompts || []);
      const total = summary.total_entries ?? entries.length;

      content.innerHTML = `
        <div class="header-title">
          <h1>Agent Registry</h1>
          <p>Inventário e validações (prompts/tools) do sistema</p>
        </div>

        <div class="dashboard-grid">
          <div class="card stat-card">
            <span class="stat-label">Total</span>
            <div class="stat-value">${total}</div>
          </div>
          <div class="card stat-card">
            <span class="stat-label">Prompts faltando</span>
            <div class="stat-value" style="color:${missingPrompts.length ? 'var(--error)' : 'var(--success)'}">${missingPrompts.length}</div>
          </div>
          <div class="card stat-card">
            <span class="stat-label">Native Agents</span>
            <div class="stat-value">${summary.count_native ?? 0}</div>
          </div>
          <div class="card stat-card">
            <span class="stat-label">Specialists</span>
            <div class="stat-value">${summary.count_specialist ?? 0}</div>
          </div>
          <div class="card stat-card">
            <span class="stat-label">LangGraph</span>
            <div class="stat-value">${summary.count_langgraph ?? 0}</div>
          </div>
        </div>

        ${missingPrompts.length ? `
          <div class="card">
            <div class="card-title">Prompts ausentes</div>
            <div class="data-list">
              ${missingPrompts.map(p => `
                <div class="data-item">
                  <div class="data-item-info">
                    <span class="data-item-title">${p}</span>
                    <span class="data-item-meta">Defina esse prompt em resync/prompts/agent_prompts.yaml</span>
                  </div>
                  <span class="badge badge-error">missing</span>
                </div>
              `).join('')}
            </div>
          </div>
        ` : ''}

        <div class="card">
          <div class="card-title">Entries</div>
          <div class="data-list">
            ${entries.map(e => {
              const v = e.validations || {};
              const ok = !!v.ok;
              const details = v.details ? JSON.stringify(v.details) : '';
              const tools = (e.tools || []).join(', ');
              const prompt = e.prompt_id || '-';
              const model = e.model || '-';
              return `
                <div class="data-item">
                  <div class="data-item-info">
                    <span class="data-item-title">${e.id} <small style="color:var(--text-secondary)">(${e.kind})</small></span>
                    <span class="data-item-meta">Model: ${model}</span>
                    <span class="data-item-meta">Prompt: ${prompt}</span>
                    <span class="data-item-meta">Tools: ${tools || '-'}</span>
                    ${details ? `<span class="data-item-meta" style="font-family:monospace; white-space:pre-wrap;">${details}</span>` : ''}
                  </div>
                  <span class="badge ${ok ? 'badge-success' : 'badge-error'}">${ok ? 'ok' : 'issues'}</span>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      `;
    } catch (err) {
      content.innerHTML = `<div class="card"><div class="stat-label">Erro ao carregar registry: ${err.message}</div></div>`;
    }
  }
}
