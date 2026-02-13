import React, { useState, useEffect } from 'react';
import { Zap, Key, Trash2, Eye, EyeOff, Copy, CheckCircle, AlertTriangle, Plus, Shield, Activity, TrendingUp } from 'lucide-react';

/**
 * Admin Interface - API Key Management
 * 
 * Design Theme: "Neumorphic Soft UI Design" (compatibility with legacy admin)
 * - Soft light background
 * - Subtle shadows (convex / pressed)
 * - Muted typography + gentle accents
 * - Optional dark mode via .dark-mode class
 */

const APIKeyAdminPanel = () => {
  const [keys, setKeys] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [createdKey, setCreatedKey] = useState(null);
  const [copiedId, setCopiedId] = useState(null);

  useEffect(() => {
    fetchKeys();
    fetchStats();
  }, []);

  const fetchKeys = async () => {
    try {
      const response = await fetch('/api/v1/admin/api-keys', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('admin_token')}`
        }
      });
      const data = await response.json();
      setKeys(data.keys);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch keys:', error);
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/v1/admin/api-keys/stats/summary', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('admin_token')}`
        }
      });
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  const createKey = async (formData) => {
    try {
      const response = await fetch('/api/v1/admin/api-keys', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('admin_token')}`
        },
        body: JSON.stringify(formData)
      });
      const data = await response.json();
      setCreatedKey(data);
      fetchKeys();
      fetchStats();
    } catch (error) {
      console.error('Failed to create key:', error);
    }
  };

  const revokeKey = async (keyId) => {
    if (!confirm('Revoke this API key? This cannot be undone.')) return;
    
    try {
      await fetch(`/api/v1/admin/api-keys/${keyId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('admin_token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ reason: 'Revoked via admin panel' })
      });
      fetchKeys();
      fetchStats();
    } catch (error) {
      console.error('Failed to revoke key:', error);
    }
  };

  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="min-h-screen overflow-hidden neu-root">
      <style jsx>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Neumorphic theme vars (aligned with legacy static/css/admin.css) */
        .neu-root {
          --bg-base: #e0e5ec;
          --bg-light: #f0f5fc;
          --bg-dark: #d1d9e6;

          --shadow-light: #ffffff;
          --shadow-dark: #a3b1c6;

          --neu-convex: 8px 8px 16px var(--shadow-dark), -8px -8px 16px var(--shadow-light);
          --neu-concave: inset 6px 6px 12px var(--shadow-dark), inset -6px -6px 12px var(--shadow-light);
          --neu-soft: 5px 5px 10px var(--shadow-dark), -5px -5px 10px var(--shadow-light);
          --neu-pressed: inset 4px 4px 8px var(--shadow-dark), inset -4px -4px 8px var(--shadow-light);

          --text-primary: #31344b;
          --text-secondary: #5a5c69;
          --text-muted: #8c8e97;

          --accent-primary: #6366f1;
          --accent-secondary: #10b981;
          --accent-warning: #f59e0b;
          --accent-danger: #ef4444;
          --accent-info: #3b82f6;

          background: var(--bg-base);
          color: var(--text-primary);
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
          -webkit-font-smoothing: antialiased;
        }

        /* Kill cyberpunk visuals */
        .neu-root :global(.cyber-grid) { display: none !important; }
        .neu-root :global(.scanline::after) { display: none !important; }
        .neu-root :global(.ascii-border::before),
        .neu-root :global(.ascii-border::after) { content: none !important; }

        /* Map common Tailwind utility colors to neumorphic palette */
        .neu-root :global(.text-cyan-50),
        .neu-root :global(.text-cyan-300),
        .neu-root :global(.text-cyan-400),
        .neu-root :global(.text-cyan-500) {
          color: var(--text-primary) !important;
        }
        .neu-root :global(.text-cyan-600\/70),
        .neu-root :global(.text-cyan-600),
        .neu-root :global(.text-slate-400) {
          color: var(--text-secondary) !important;
        }

        .neu-root :global(.text-green-400) { color: var(--accent-secondary) !important; }
        .neu-root :global(.text-red-400) { color: var(--accent-danger) !important; }
        .neu-root :global(.text-amber-400) { color: var(--accent-warning) !important; }

        /* Backgrounds */
        .neu-root :global(.bg-slate-900),
        .neu-root :global(.bg-slate-900\/50),
        .neu-root :global(.bg-black\/80),
        .neu-root :global(.bg-black\/50),
        .neu-root :global(.bg-gradient-to-r),
        .neu-root :global(.bg-gradient-to-br) {
          background: var(--bg-base) !important;
        }

        .neu-root :global(.bg-cyan-500\/20),
        .neu-root :global(.bg-cyan-500\/10),
        .neu-root :global(.bg-green-500\/10),
        .neu-root :global(.bg-red-500\/10),
        .neu-root :global(.bg-amber-500\/10),
        .neu-root :global(.bg-purple-500\/20) {
          background: var(--bg-light) !important;
        }

        /* Borders -> subtle */
        .neu-root :global([class*="border-"]) {
          border-color: rgba(49, 52, 75, 0.12) !important;
        }

        /* Card / panel look */
        .neu-root :global(.glow-border),
        .neu-root :global(.holographic) {
          box-shadow: var(--neu-soft) !important;
          border: none !important;
          border-radius: 16px !important;
          background: var(--bg-base) !important;
        }

        /* Header */
        .neu-root header {
          background: var(--bg-base) !important;
          border-bottom: 1px solid rgba(49, 52, 75, 0.10) !important;
          box-shadow: var(--neu-soft);
        }

        /* Buttons */
        .neu-root button {
          border-radius: 14px;
        }
        .neu-root button:not([disabled]) {
          background: var(--bg-base) !important;
          color: var(--text-primary) !important;
          box-shadow: var(--neu-soft) !important;
          border: none !important;
        }
        .neu-root button:hover {
          box-shadow: var(--neu-convex) !important;
        }
        .neu-root button:active {
          box-shadow: var(--neu-pressed) !important;
        }

        /* Code chips */
        .neu-root :global(code) {
          background: var(--bg-light) !important;
          border-radius: 12px;
          padding: 6px 10px;
          box-shadow: var(--neu-pressed);
          color: var(--text-primary) !important;
        }
      `}</style>

      {/* Grid background */}
      <div className="fixed inset-0 cyber-grid pointer-events-none"></div>

      {/* Header */}
      <header className="relative z-10 border-b border-cyan-900/50 bg-gradient-to-r from-slate-900/50 to-slate-800/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-2 bg-cyan-500/20 rounded-lg glow-border">
                <Shield className="w-8 h-8 text-cyan-400" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-cyan-400 tracking-wider">
                  <span className="text-cyan-600">[</span>API_KEY_MANAGER<span className="text-cyan-600">]</span>
                </h1>
                <p className="text-cyan-600/70 text-sm mt-1">// Security Access Control System</p>
              </div>
            </div>
            <button
              onClick={() => setShowCreateModal(true)}
              className="px-6 py-3 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/50 rounded-lg flex items-center space-x-2 transition-all glow-border group"
            >
              <Plus className="w-5 h-5 group-hover:rotate-90 transition-transform" />
              <span className="font-semibold">NEW_KEY()</span>
            </button>
          </div>
        </div>
      </header>

      <main className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        {/* Stats Grid */}
        {stats && (
          <div className="grid grid-cols-4 gap-6 mb-8 fade-in-up">
            <StatCard
              icon={<Key className="w-6 h-6" />}
              label="TOTAL_KEYS"
              value={stats.counts.total}
              color="cyan"
            />
            <StatCard
              icon={<Activity className="w-6 h-6 status-active" />}
              label="ACTIVE"
              value={stats.counts.active}
              color="green"
            />
            <StatCard
              icon={<AlertTriangle className="w-6 h-6" />}
              label="REVOKED"
              value={stats.counts.revoked}
              color="red"
            />
            <StatCard
              icon={<TrendingUp className="w-6 h-6" />}
              label="EXPIRED"
              value={stats.counts.expired}
              color="amber"
            />
          </div>
        )}

        {/* Keys List */}
        <div className="space-y-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-cyan-400">
              <span className="text-cyan-600">{'> '}</span>REGISTERED_KEYS
            </h2>
            <span className="text-cyan-600/70 text-sm">{keys.length} entries</span>
          </div>

          {loading ? (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400"></div>
              <p className="mt-4 text-cyan-600">Loading...</p>
            </div>
          ) : (
            <div className="space-y-3">
              {keys.map((key, index) => (
                <KeyCard
                  key={key.id}
                  keyData={key}
                  onRevoke={revokeKey}
                  onCopy={copyToClipboard}
                  copied={copiedId === key.id}
                  index={index}
                />
              ))}
            </div>
          )}
        </div>
      </main>

      {/* Create Modal */}
      {showCreateModal && (
        <CreateKeyModal
          onClose={() => {
            setShowCreateModal(false);
            setCreatedKey(null);
          }}
          onCreate={createKey}
          createdKey={createdKey}
          onCopy={copyToClipboard}
        />
      )}
    </div>
  );
};

// Stat Card Component
const StatCard = ({ icon, label, value, color }) => {
  const colorClasses = {
    cyan: 'from-cyan-500/20 to-cyan-600/20 border-cyan-500/50 text-cyan-400',
    green: 'from-green-500/20 to-green-600/20 border-green-500/50 text-green-400',
    red: 'from-red-500/20 to-red-600/20 border-red-500/50 text-red-400',
    amber: 'from-amber-500/20 to-amber-600/20 border-amber-500/50 text-amber-400'
  };

  return (
    <div className={`p-6 rounded-lg border bg-gradient-to-br backdrop-blur-sm holographic fade-in-up ${colorClasses[color]}`}>
      <div className="flex items-center justify-between mb-3">
        {icon}
        <span className="text-3xl font-bold">{value}</span>
      </div>
      <div className="text-sm opacity-70 font-medium">{label}</div>
    </div>
  );
};

// Key Card Component
const KeyCard = ({ keyData, onRevoke, onCopy, copied, index }) => {
  const [expanded, setExpanded] = useState(false);

  const statusColor = keyData.is_valid 
    ? 'text-green-400 bg-green-500/20 border-green-500/50' 
    : keyData.is_revoked 
    ? 'text-red-400 bg-red-500/20 border-red-500/50'
    : 'text-amber-400 bg-amber-500/20 border-amber-500/50';

  const statusText = keyData.is_valid 
    ? 'ACTIVE' 
    : keyData.is_revoked 
    ? 'REVOKED' 
    : 'EXPIRED';

  return (
    <div 
      className="p-5 bg-slate-900/50 border border-cyan-900/30 rounded-lg hover:border-cyan-500/50 transition-all backdrop-blur-sm scanline relative overflow-hidden fade-in-up"
      style={{ animationDelay: `${index * 50}ms` }}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-3 mb-3">
            <Key className="w-5 h-5 text-cyan-400" />
            <h3 className="text-lg font-semibold text-cyan-300">{keyData.name}</h3>
            <span className={`text-xs px-3 py-1 rounded-full border font-semibold ${statusColor}`}>
              {statusText}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-4 mb-3 text-sm">
            <div>
              <span className="text-cyan-600/70">PREFIX:</span>
              <div className="flex items-center space-x-2 mt-1">
                <code className="text-cyan-400 bg-cyan-500/10 px-2 py-1 rounded">{keyData.key_prefix}...</code>
                <button
                  onClick={() => onCopy(keyData.key_prefix, keyData.id)}
                  className="p-1 hover:bg-cyan-500/20 rounded transition-colors"
                >
                  {copied ? <CheckCircle className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-cyan-500" />}
                </button>
              </div>
            </div>
            <div>
              <span className="text-cyan-600/70">USAGE:</span>
              <p className="text-cyan-400 mt-1">{keyData.usage_count.toLocaleString()} calls</p>
            </div>
            <div>
              <span className="text-cyan-600/70">CREATED:</span>
              <p className="text-cyan-400 mt-1">{new Date(keyData.created_at).toLocaleDateString()}</p>
            </div>
          </div>

          {keyData.description && (
            <p className="text-cyan-600/70 text-sm mb-3">// {keyData.description}</p>
          )}

          <div className="flex items-center space-x-2 text-sm">
            <span className="text-cyan-600/70">SCOPES:</span>
            {keyData.scopes.map(scope => (
              <span key={scope} className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded text-xs border border-purple-500/30">
                {scope}
              </span>
            ))}
          </div>
        </div>

        <div className="flex space-x-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="p-2 hover:bg-cyan-500/20 rounded transition-colors text-cyan-400"
            title="Details"
          >
            {expanded ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
          </button>
          {keyData.is_valid && (
            <button
              onClick={() => onRevoke(keyData.id)}
              className="p-2 hover:bg-red-500/20 rounded transition-colors text-red-400"
              title="Revoke"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-cyan-900/30 space-y-2 text-sm">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <span className="text-cyan-600/70">ID:</span>
              <p className="text-cyan-400 font-mono text-xs mt-1">{keyData.id}</p>
            </div>
            <div>
              <span className="text-cyan-600/70">CREATED_BY:</span>
              <p className="text-cyan-400 mt-1">{keyData.created_by}</p>
            </div>
            {keyData.last_used_at && (
              <div>
                <span className="text-cyan-600/70">LAST_USED:</span>
                <p className="text-cyan-400 mt-1">{new Date(keyData.last_used_at).toLocaleString()}</p>
              </div>
            )}
            {keyData.expires_at && (
              <div>
                <span className="text-cyan-600/70">EXPIRES:</span>
                <p className={`mt-1 ${keyData.is_expired ? 'text-red-400' : 'text-cyan-400'}`}>
                  {new Date(keyData.expires_at).toLocaleString()}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// Create Key Modal Component
const CreateKeyModal = ({ onClose, onCreate, createdKey, onCopy }) => {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    scopes: ['metrics:write'],
    expires_in_days: 365
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onCreate(formData);
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 border-2 border-cyan-500/50 rounded-lg max-w-2xl w-full glow-border">
        <div className="p-6 border-b border-cyan-900/50">
          <h2 className="text-2xl font-bold text-cyan-400">
            <span className="text-cyan-600">{'> '}</span>CREATE_NEW_KEY()
          </h2>
        </div>

        {createdKey ? (
          <div className="p-6 space-y-4">
            <div className="p-4 bg-green-500/10 border border-green-500/50 rounded-lg">
              <div className="flex items-start space-x-3">
                <CheckCircle className="w-6 h-6 text-green-400 flex-shrink-0 mt-1" />
                <div className="flex-1">
                  <p className="text-green-400 font-semibold mb-2">Key Created Successfully!</p>
                  <p className="text-cyan-600/70 text-sm mb-4">⚠️ Copy this key now - you won't see it again!</p>
                  <div className="bg-black/50 p-4 rounded border border-cyan-500/30 font-mono">
                    <div className="flex items-center justify-between">
                      <code className="text-cyan-400 text-sm break-all">{createdKey.key}</code>
                      <button
                        onClick={() => onCopy(createdKey.key, 'created')}
                        className="ml-4 p-2 bg-cyan-500/20 hover:bg-cyan-500/30 rounded transition-colors flex-shrink-0"
                      >
                        <Copy className="w-5 h-5 text-cyan-400" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <button
              onClick={onClose}
              className="w-full py-3 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/50 rounded transition-all font-semibold"
            >
              CLOSE
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="p-6 space-y-4">
            <div>
              <label className="block text-cyan-400 mb-2 font-semibold">NAME *</label>
              <input
                type="text"
                required
                value={formData.name}
                onChange={(e) => setFormData({...formData, name: e.target.value})}
                className="w-full px-4 py-3 bg-black/50 border border-cyan-500/30 rounded text-cyan-300 focus:border-cyan-500 focus:outline-none transition-colors"
                placeholder="e.g., Production FTA Metrics"
              />
            </div>

            <div>
              <label className="block text-cyan-400 mb-2 font-semibold">DESCRIPTION</label>
              <textarea
                value={formData.description}
                onChange={(e) => setFormData({...formData, description: e.target.value})}
                className="w-full px-4 py-3 bg-black/50 border border-cyan-500/30 rounded text-cyan-300 focus:border-cyan-500 focus:outline-none transition-colors"
                rows="3"
                placeholder="Optional description..."
              />
            </div>

            <div>
              <label className="block text-cyan-400 mb-2 font-semibold">SCOPES</label>
              <select
                multiple
                value={formData.scopes}
                onChange={(e) => setFormData({...formData, scopes: Array.from(e.target.selectedOptions, option => option.value)})}
                className="w-full px-4 py-3 bg-black/50 border border-cyan-500/30 rounded text-cyan-300 focus:border-cyan-500 focus:outline-none transition-colors"
              >
                <option value="metrics:read">metrics:read</option>
                <option value="metrics:write">metrics:write</option>
                <option value="admin:read">admin:read</option>
                <option value="admin:write">admin:write</option>
                <option value="workflows:read">workflows:read</option>
                <option value="workflows:write">workflows:write</option>
              </select>
            </div>

            <div>
              <label className="block text-cyan-400 mb-2 font-semibold">EXPIRES_IN (days)</label>
              <input
                type="number"
                value={formData.expires_in_days}
                onChange={(e) => setFormData({...formData, expires_in_days: parseInt(e.target.value)})}
                className="w-full px-4 py-3 bg-black/50 border border-cyan-500/30 rounded text-cyan-300 focus:border-cyan-500 focus:outline-none transition-colors"
                min="1"
                max="3650"
              />
            </div>

            <div className="flex space-x-3 pt-4">
              <button
                type="submit"
                className="flex-1 py-3 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/50 rounded transition-all font-semibold"
              >
                CREATE_KEY()
              </button>
              <button
                type="button"
                onClick={onClose}
                className="flex-1 py-3 bg-slate-700/50 hover:bg-slate-700/70 border border-slate-600/50 rounded transition-all font-semibold"
              >
                CANCEL
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

export default APIKeyAdminPanel;
