/**
 * Admin Advanced Module
 * Surface legacy /api/admin endpoints moved/aliased into /admin namespace.
 */
class AdminAdvanced {
  constructor(app) {
    this.app = app;
    this.api = app.api;
  }

  async loadAdvancedView() {
    const content = document.getElementById('content');
    content.innerHTML = '<div class="card"><div class="stat-label">Carregando Advanced Admin APIs...</div></div>';

    const sections = [];

    // GraphRAG
    try {
      const stats = await this.api.get('/admin/graphrag/stats');
      sections.push({
        title: 'GraphRAG Admin',
        badge: 'ok',
        body: `<pre style="white-space:pre-wrap;">${this._safeJson(stats)}</pre>`
      });
    } catch (e) {
      sections.push({
        title: 'GraphRAG Admin',
        badge: 'error',
        body: `<div class="stat-label">Falha ao carregar /admin/graphrag/stats: ${e.message}</div>`
      });
    }

    // Document KG
    try {
      const kgStats = await this.api.get('/admin/kg/stats');
      sections.push({
        title: 'Document KG Admin',
        badge: 'ok',
        body: `<pre style="white-space:pre-wrap;">${this._safeJson(kgStats)}</pre>`
      });
    } catch (e) {
      sections.push({
        title: 'Document KG Admin',
        badge: 'error',
        body: `<div class="stat-label">Falha ao carregar /admin/kg/stats: ${e.message}</div>`
      });
    }

    // Unified Config API status
    try {
      const cfgStatus = await this.api.get('/admin/config-api/status');
      sections.push({
        title: 'Unified Config API',
        badge: 'ok',
        body: `<pre style="white-space:pre-wrap;">${this._safeJson(cfgStatus)}</pre>`
      });
    } catch (e) {
      sections.push({
        title: 'Unified Config API',
        badge: 'error',
        body: `<div class="stat-label">Falha ao carregar /admin/config-api/status: ${e.message}</div>`
      });
    }

    content.innerHTML = `
      <div class="header-title">
        <h1>Advanced Admin APIs</h1>
        <p>Visão rápida de endpoints administrativos que antes estavam em /api/admin</p>
      </div>
      ${sections.map(s => `
        <div class="card">
          <div class="card-title">
            ${s.title}
            <span class="badge ${s.badge === 'ok' ? 'badge-success' : 'badge-error'}" style="margin-left: 10px;">${s.badge}</span>
          </div>
          ${s.body}
        </div>
      `).join('')}
    `;
  }

  _safeJson(obj) {
    try { return JSON.stringify(obj, null, 2); } catch { return String(obj); }
  }
}
