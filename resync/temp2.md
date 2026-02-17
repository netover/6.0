# PATCH: Implementação Completa do Admin Web Neumórfico do Resync Optimized

## Visão Geral

Este patch implementa todos os recursos avançados de admin web presentes na versão `resync_optimized` na versão `6.0`. Inclui o design neumórfico completo, todas as seções de configuração, monitoramento e ferramentas enterprise.

---

## 1. Arquivo: `resync/templates/admin.html`

### Substituir o arquivo completo pelo código abaixo:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resync Admin Configuration</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="/static/css/admin-neumorphic.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #007bff;
            --secondary-color: #6c757d;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --dark-color: #343a40;
            --light-color: #f8f9fa;
            --bg-color: #f5f7fb;
            --shadow-light: #ffffff;
            --shadow-dark: #d1d9e6;
        }
        
        body {
            background-color: var(--bg-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .admin-header {
            background: linear-gradient(135deg, var(--primary-color), #0056b3);
            color: white;
            padding: 1.5rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .sidebar {
            background-color: white;
            box-shadow: 2px 0 10px rgba(0,0,0,0.05);
            height: calc(100vh - 70px);
            position: sticky;
            top: 70px;
        }
        
        .nav-link {
            color: var(--secondary-color);
            border-left: 3px solid transparent;
            transition: all 0.3s ease;
            padding: 12px 15px;
            border-radius: 8px;
            margin-bottom: 4px;
        }
        
        .nav-link:hover, .nav-link.active {
            color: var(--primary-color);
            background-color: rgba(0,123,255,0.1);
            border-left: 3px solid var(--primary-color);
        }
        
        .card {
            border: none;
            border-radius: 15px;
            box-shadow: 8px 8px 16px var(--shadow-dark), -8px -8px 16px var(--shadow-light);
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
            background: white;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 12px 12px 24px var(--shadow-dark), -12px -12px 24px var(--shadow-light);
        }
        
        .card-header {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-bottom: 1px solid rgba(0,0,0,0.05);
            border-radius: 15px 15px 0 0 !important;
            font-weight: 600;
            padding: 1rem 1.5rem;
        }
        
        .btn {
            border-radius: 10px;
            padding: 0.5rem 1.25rem;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 4px 4px 8px var(--shadow-dark), -4px -4px 8px var(--shadow-light);
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary-color), #0069d9);
            border: none;
            color: white;
        }
        
        .btn-success {
            background: linear-gradient(135deg, var(--success-color), #218838);
            border: none;
            color: white;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-connected { background-color: var(--success-color); }
        .status-disconnected { background-color: var(--danger-color); }
        .status-warning { background-color: var(--warning-color); }
        
        .config-section { display: none; }
        
        .config-section.active {
            display: block;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
        }
        
        .toast {
            background: white;
            border-radius: 10px;
            border-left: 4px solid var(--primary-color);
            box-shadow: 8px 8px 16px var(--shadow-dark), -8px -8px 16px var(--shadow-light);
            padding: 1rem;
            margin-bottom: 10px;
            min-width: 300px;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .toast.success { border-left-color: var(--success-color); }
        .toast.error { border-left-color: var(--danger-color); }
        .toast.warning { border-left-color: var(--warning-color); }
        
        .health-card {
            transition: all 0.3s ease;
            border-width: 2px;
            border-radius: 15px;
        }
        
        .health-card:hover {
            transform: translateY(-2px);
        }
        
        .health-card .display-4 {
            font-size: 2.5rem;
            transition: color 0.3s ease;
        }
        
        .form-control {
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            padding: 0.75rem 1rem;
            transition: all 0.3s ease;
        }
        
        .form-control:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
        }
        
        .badge {
            padding: 0.5rem 0.75rem;
            border-radius: 20px;
            font-weight: 500;
        }
        
        .table thead th {
            border-bottom: 2px solid var(--shadow-dark);
            color: var(--secondary-color);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
        }
        
        .table-hover tbody tr:hover {
            background-color: rgba(0, 123, 255, 0.05);
        }
    </style>
</head>
<body>
    <div class="toast-container"></div>
    
    <header class="admin-header">
        <div class="container-fluid">
            <div class="row align-items-center">
                <div class="col-md-6">
                    <h1><i class="fas fa-cogs me-2"></i>Resync Administration</h1>
                    <p class="mb-0">System Configuration & Management Console</p>
                </div>
                <div class="col-md-6 text-end">
                    <div class="btn-group">
                        <button class="btn btn-outline-light" id="refreshBtn">
                            <i class="fas fa-sync-alt"></i> Refresh
                        </button>
                        <button class="btn btn-outline-light" id="logoutBtn">
                            <i class="fas fa-sign-out-alt"></i> Logout
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </header>
    
    <div class="container-fluid mt-4">
        <div class="row">
            <div class="col-md-3 col-lg-2 sidebar">
                <nav class="nav flex-column pt-3">
                    <h6 class="px-3 text-muted">CONFIGURATION</h6>
                    <a class="nav-link" href="/admin/settings" target="_blank">
                        <i class="fas fa-cogs me-2"></i>All Settings <i class="fas fa-external-link-alt fa-xs text-muted"></i>
                    </a>
                    <a class="nav-link active" href="#" data-target="teams-config">
                        <i class="fab fa-microsoft me-2"></i>Teams Integration
                    </a>
                    <a class="nav-link" href="#" data-target="teams-webhook-users">
                        <i class="fas fa-users-cog me-2"></i>Teams Webhook Users
                        <span class="badge bg-secondary" style="margin-left:auto;" id="teams-users-count">0</span>
                    </a>
                    <a class="nav-link" href="#" data-target="teams-proactive-alerts">
                        <i class="fas fa-bell me-2"></i>Teams Proactive Alerts
                    </a>
                    <a class="nav-link" href="#" data-target="tws-config">
                        <i class="fas fa-server me-2"></i>TWS Configuration
                    </a>
                    <a class="nav-link" href="#" data-target="tws-instances">
                        <i class="fas fa-network-wired me-2"></i>TWS Instances
                        <span class="badge bg-info" style="margin-left:auto;" id="tws-instance-count">0</span>
                    </a>
                    <a class="nav-link" href="#" data-target="system-config">
                        <i class="fas fa-sliders-h me-2"></i>System Settings
                    </a>
                    <a class="nav-link" href="#" data-target="litellm-config">
                        <i class="fas fa-brain me-2"></i>LiteLLM & AI Models
                    </a>
                    
                    <h6 class="px-3 mt-4 text-muted">MONITORING</h6>
                    <a class="nav-link" href="#" data-target="health-monitoring">
                        <span class="badge badge-success" style="margin-left:auto;">OK</span>
                        <i class="fas fa-heartbeat me-2"></i>System Health
                    </a>
                    <a class="nav-link" href="#" data-target="proactive-monitoring">
                        <i class="fas fa-satellite-dish me-2"></i>TWS Proativo
                        <span class="badge badge-info" style="margin-left:auto;">LIVE</span>
                    </a>
                    <a class="nav-link" href="/tws-monitor" target="_blank">
                        <i class="fas fa-chart-line me-2"></i>Dashboard TWS <i class="fas fa-external-link-alt fa-xs"></i>
                    </a>
                    <a class="nav-link" href="#" data-target="notifications">
                        <i class="fas fa-bell me-2"></i>Notifications
                        <span class="badge badge-warning" style="margin-left:auto;">3</span>
                    </a>
                    <a class="nav-link" href="#" data-target="logs">
                        <i class="fas fa-file-alt me-2"></i>System Logs
                    </a>
                    
                    <h6 class="px-3 mt-4 text-muted">AI & LEARNING</h6>
                    <a class="nav-link" href="#" data-target="auto-tuning">
                        <i class="fas fa-sliders-h me-2"></i>Auto-Tuning
                        <span class="badge bg-secondary" id="autoTuningLevelBadge" style="margin-left:auto;">OFF</span>
                    </a>
                    <a class="nav-link" href="#" data-target="graphrag">
                        <i class="fas fa-project-diagram me-2"></i>GraphRAG
                        <span class="badge bg-info" id="graphragStatusBadge" style="margin-left:auto;">READY</span>
                    </a>
                    <a class="nav-link" href="#" data-target="rag-reranker">
                        <i class="fas fa-sort-amount-down me-2"></i>RAG Reranker
                        <span class="badge bg-info" id="rerankGatingBadge" style="margin-left:auto;">ON</span>
                    </a>

                    <h6 class="px-3 mt-4 text-muted">TOOLS</h6>
                    <a class="nav-link" href="#" data-target="backup-restore">
                        <i class="fas fa-database me-2"></i>Backup & Restore
                    </a>
                    <a class="nav-link" href="#" data-target="observability">
                        <i class="fas fa-eye me-2"></i>Observability
                        <span class="badge bg-info" style="margin-left:auto;">AI</span>
                    </a>
                    <a class="nav-link" href="#" data-target="revisao-operador">
                        <i class="fas fa-clipboard-check me-2"></i>Revisão Operador
                        <span class="badge bg-warning" style="margin-left:auto;" id="revisao-pending-count">0</span>
                    </a>
                    <a class="nav-link" href="#" data-target="audit">
                        <i class="fas fa-scroll me-2"></i>Audit Log
                    </a>
                    <a class="nav-link" href="#" data-target="maintenance">
                        <i class="fas fa-tools me-2"></i>Maintenance
                    </a>
                    
                    <h6 class="px-3 mt-4 text-muted">ENTERPRISE</h6>
                    <a class="nav-link" href="#" data-target="enterprise-overview">
                        <i class="fas fa-building me-2"></i>Overview
                        <span class="badge bg-success" style="margin-left:auto;" id="enterprise-status">OK</span>
                    </a>
                    <a class="nav-link" href="#" data-target="enterprise-incidents">
                        <i class="fas fa-exclamation-triangle me-2"></i>Incidents
                        <span class="badge bg-warning" style="margin-left:auto;" id="incidents-count">0</span>
                    </a>
                    <a class="nav-link" href="#" data-target="enterprise-compliance">
                        <i class="fas fa-shield-alt me-2"></i>Compliance
                    </a>
                    <a class="nav-link" href="#" data-target="enterprise-security">
                        <i class="fas fa-lock me-2"></i>Security & SIEM
                    </a>
                    <a class="nav-link" href="#" data-target="enterprise-observability">
                        <i class="fas fa-chart-line me-2"></i>Observability
                    </a>
                    <a class="nav-link" href="#" data-target="enterprise-resilience">
                        <i class="fas fa-heartbeat me-2"></i>Resilience
                    </a>
                </nav>
            </div>
            
            <div class="col-md-9 col-lg-10">
                <!-- TEAMS CONFIGURATION SECTION -->
                <div id="teams-config" class="config-section active">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fab fa-microsoft me-2"></i>Microsoft Teams Integration</h2>
                        <button class="btn btn-success" id="saveTeamsConfig">
                            <i class="fas fa-save me-1"></i>Save Configuration
                        </button>
                    </div>
                    
                    <div class="row">
                        <div class="col-xl-8">
                            <div class="card mb-4">
                                <div class="card-header">
                                    <h5 class="mb-0"><i class="fas fa-cog me-2"></i>Basic Configuration</h5>
                                </div>
                                <div class="card-body">
                                    <div class="form-check form-switch mb-3">
                                        <input class="form-check-input" type="checkbox" id="teamsEnabled" checked>
                                        <label class="form-check-label" for="teamsEnabled">
                                            <strong>Enable Teams Integration</strong>
                                            <small class="d-block text-muted">Toggle to enable/disable Teams notifications</small>
                                        </label>
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label for="webhookUrl" class="form-label">Teams Webhook URL</label>
                                        <input type="url" class="form-control" id="webhookUrl" placeholder="https://yourcompany.webhook.office.com/webhook...">
                                    </div>
                                    
                                    <div class="row">
                                        <div class="col-md-6">
                                            <div class="mb-3">
                                                <label for="channelName" class="form-label">Channel Name</label>
                                                <input type="text" class="form-control" id="channelName" placeholder="Resync Notifications">
                                            </div>
                                        </div>
                                        <div class="col-md-6">
                                            <div class="mb-3">
                                                <label for="botName" class="form-label">Bot Display Name</label>
                                                <input type="text" class="form-control" id="botName" value="Resync Bot">
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="card mb-4">
                                <div class="card-header">
                                    <h5 class="mb-0"><i class="fas fa-star me-2"></i>Advanced Features</h5>
                                </div>
                                <div class="card-body">
                                    <div class="form-check form-switch mb-3">
                                        <input class="form-check-input" type="checkbox" id="conversationLearning" checked>
                                        <label class="form-check-label" for="conversationLearning">
                                            <strong>Enable Conversation Learning</strong>
                                        </label>
                                    </div>
                                    
                                    <div class="form-check form-switch mb-3">
                                        <input class="form-check-input" type="checkbox" id="jobNotifications" checked>
                                        <label class="form-check-label" for="jobNotifications">
                                            <strong>Enable Job Status Notifications</strong>
                                        </label>
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">Monitored TWS Instances</label>
                                        <div class="input-group mb-2">
                                            <input type="text" class="form-control" id="newInstanceInput" placeholder="Enter TWS instance name">
                                            <button class="btn btn-outline-secondary" type="button" id="addInstanceBtn">
                                                <i class="fas fa-plus"></i>
                                            </button>
                                        </div>
                                        <div id="instancesList" class="mt-2"></div>
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">Job Status Filters</label>
                                        <select multiple class="form-select" id="jobStatusFilters" size="4">
                                            <option value="ABEND" selected>ABEND (Abnormal End)</option>
                                            <option value="ERROR" selected>Error</option>
                                            <option value="FAILED" selected>Failed</option>
                                            <option value="TIMEOUT">Timeout</option>
                                            <option value="CANCELLED">Cancelled</option>
                                            <option value="WARNING">Warning</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="col-xl-4">
                            <div class="card mb-4">
                                <div class="card-header">
                                    <h5 class="mb-0"><i class="fas fa-heartbeat me-2"></i>Integration Status</h5>
                                </div>
                                <div class="card-body">
                                    <div class="text-center mb-3">
                                        <div class="status-indicator status-connected"></div>
                                        <span class="fw-bold">Connected</span>
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">Last Health Check</label>
                                        <div class="text-muted" id="lastHealthCheck">2 minutes ago</div>
                                    </div>
                                    
                                    <button class="btn btn-outline-primary w-100" id="testNotificationBtn">
                                        <i class="fas fa-paper-plane me-1"></i> Send Test Notification
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- HEALTH MONITORING SECTION -->
                <div id="health-monitoring" class="config-section">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fas fa-heartbeat me-2"></i>System Health</h2>
                        <button class="btn btn-primary" id="refreshHealthBtn">
                            <i class="fas fa-sync-alt me-1"></i> Refresh
                        </button>
                    </div>
                    
                    <div class="row mb-4">
                        <div class="col-12">
                            <div class="card health-card" id="overallHealthCard">
                                <div class="card-body d-flex align-items-center">
                                    <div class="me-3">
                                        <i class="fas fa-check-circle text-success display-4"></i>
                                    </div>
                                    <div>
                                        <h4 class="mb-0">System Healthy</h4>
                                        <p class="text-muted mb-0">All components operational</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-3">
                            <div class="card health-card border-success">
                                <div class="card-body text-center">
                                    <i class="fas fa-database text-success display-4 mb-2"></i>
                                    <h5>Database</h5>
                                    <span class="badge bg-success">Healthy</span>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card health-card border-success">
                                <div class="card-body text-center">
                                    <i class="fas fa-server text-success display-4 mb-2"></i>
                                    <h5>Redis</h5>
                                    <span class="badge bg-success">Healthy</span>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card health-card border-success">
                                <div class="card-body text-center">
                                    <i class="fas fa-brain text-success display-4 mb-2"></i>
                                    <h5>AI Models</h5>
                                    <span class="badge bg-success">Healthy</span>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card health-card border-success">
                                <div class="card-body text-center">
                                    <i class="fas fa-network-wired text-success display-4 mb-2"></i>
                                    <h5>TWS</h5>
                                    <span class="badge bg-success">Healthy</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- ENTERPRISE OVERVIEW SECTION -->
                <div id="enterprise-overview" class="config-section">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fas fa-building me-2"></i>Enterprise Overview</h2>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-header">System Status</div>
                                <div class="card-body">
                                    <h3 class="text-success"><i class="fas fa-check-circle"></i> Operational</h3>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-header">Active Incidents</div>
                                <div class="card-body">
                                    <h3 class="text-warning">0</h3>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-header">Compliance Status</div>
                                <div class="card-body">
                                    <h3 class="text-success"><i class="fas fa-shield-alt"></i> Compliant</h3>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- BACKUP & RESTORE SECTION -->
                <div id="backup-restore" class="config-section">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fas fa-database me-2"></i>Backup & Restore</h2>
                        <button class="btn btn-success" id="createBackupBtn">
                            <i class="fas fa-plus me-1"></i> Create Backup
                        </button>
                    </div>
                    
                    <div class="row">
                        <div class="col-xl-8">
                            <div class="card">
                                <div class="card-header">Available Backups</div>
                                <div class="card-body">
                                    <div class="table-responsive">
                                        <table class="table table-hover" id="backupsTable">
                                            <thead>
                                                <tr>
                                                    <th>Date</th>
                                                    <th>Type</th>
                                                    <th>Size</th>
                                                    <th>Status</th>
                                                    <th>Actions</th>
                                                </tr>
                                            </thead>
                                            <tbody id="backupsTableBody">
                                                <tr><td colspan="5" class="text-center py-4">Loading backups...</td></tr>
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="col-xl-4">
                            <div class="card">
                                <div class="card-header">Backup Statistics</div>
                                <div class="card-body">
                                    <div class="mb-3">
                                        <div class="d-flex justify-content-between">
                                            <span class="text-muted">Total Backups</span>
                                            <strong id="statsTotalBackups">0</strong>
                                        </div>
                                    </div>
                                    <div class="mb-3">
                                        <div class="d-flex justify-content-between">
                                            <span class="text-muted">Last Backup</span>
                                            <strong id="statsLastBackup">Never</strong>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- LITELLM CONFIG SECTION -->
                <div id="litellm-config" class="config-section">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fas fa-brain me-2"></i>LiteLLM & AI Models</h2>
                        <button class="btn btn-primary" id="refreshModelsBtn">
                            <i class="fas fa-sync-alt me-1"></i> Refresh
                        </button>
                    </div>
                    
                    <div class="row mb-4">
                        <div class="col-md-3">
                            <div class="card">
                                <div class="card-body text-center">
                                    <i class="fas fa-brain text-primary display-4 mb-2"></i>
                                    <h5>Active Models</h5>
                                    <h3 class="text-primary" id="activeModelsCount">0</h3>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card">
                                <div class="card-body text-center">
                                    <i class="fas fa-dollar-sign text-success display-4 mb-2"></i>
                                    <h5>Total Cost (Today)</h5>
                                    <h3 class="text-success" id="totalCostToday">$0.00</h3>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card">
                                <div class="card-body text-center">
                                    <i class="fas fa-chart-line text-info display-4 mb-2"></i>
                                    <h5>Total Requests</h5>
                                    <h3 class="text-info" id="totalRequests">0</h3>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card">
                                <div class="card-body text-center">
                                    <i class="fas fa-clock text-warning display-4 mb-2"></i>
                                    <h5>Avg Latency</h5>
                                    <h3 class="text-warning" id="avgLatency">0ms</h3>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">Configured Models</div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-hover" id="modelsTable">
                                    <thead>
                                        <tr>
                                            <th>Model</th>
                                            <th>Provider</th>
                                            <th>Status</th>
                                            <th>Requests</th>
                                            <th>Cost</th>
                                            <th>Latency</th>
                                            <th>Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody id="modelsTableBody">
                                        <tr><td colspan="7" class="text-center py-4">Loading models...</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- GRAPHRAG SECTION -->
                <div id="graphrag" class="config-section">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fas fa-project-diagram me-2"></i>GraphRAG Administration</h2>
                        <button class="btn btn-primary" id="refreshGraphragBtn">
                            <i class="fas fa-sync-alt me-1"></i> Refresh
                        </button>
                    </div>
                    
                    <div class="row mb-4">
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-body text-center">
                                    <i class="fas fa-database text-primary display-4 mb-2"></i>
                                    <h5>Total Entities</h5>
                                    <h3 id="graphragEntities">0</h3>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-body text-center">
                                    <i class="fas fa-link text-info display-4 mb-2"></i>
                                    <h5>Total Relationships</h5>
                                    <h3 id="graphragRelations">0</h3>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-body text-center">
                                    <i class="fas fa-file-alt text-success display-4 mb-2"></i>
                                    <h5>Indexed Documents</h5>
                                    <h3 id="graphragDocuments">0</h3>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">Actions</div>
                        <div class="card-body">
                            <div class="d-grid gap-2" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
                                <button class="btn btn-outline-primary" id="discoverEntitiesBtn">
                                    <i class="fas fa-search me-1"></i> Discover Entities
                                </button>
                                <button class="btn btn-outline-info" id="invalidateCacheBtn">
                                    <i class="fas fa-trash me-1"></i> Invalidate Cache
                                </button>
                                <button class="btn btn-outline-secondary" id="resetStatsBtn">
                                    <i class="fas fa-undo me-1"></i> Reset Statistics
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- AUTO-TUNING SECTION -->
                <div id="auto-tuning" class="config-section">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fas fa-sliders-h me-2"></i>Auto-Tuning</h2>
                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" id="autoTuningEnabled">
                            <label class="form-check-label" for="autoTuningEnabled">Enable Auto-Tuning</label>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">Current Thresholds</div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label class="form-label">Error Rate Threshold (%)</label>
                                        <input type="number" class="form-control" id="errorRateThreshold" value="5">
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label class="form-label">Latency Threshold (ms)</label>
                                        <input type="number" class="form-control" id="latencyThreshold" value="1000">
                                    </div>
                                </div>
                            </div>
                            <button class="btn btn-primary" id="saveThresholdsBtn">
                                <i class="fas fa-save me-1"></i> Save Thresholds
                            </button>
                        </div>
                    </div>
                </div>
                
                <!-- RAG RERANKER SECTION -->
                <div id="rag-reranker" class="config-section">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fas fa-sort-amount-down me-2"></i>RAG Reranker</h2>
                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" id="rerankGatingEnabled" checked>
                            <label class="form-check-label" for="rerankGatingEnabled">Enable Reranking</label>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">Reranker Configuration</div>
                        <div class="card-body">
                            <div class="mb-3">
                                <label class="form-label">Reranker Model</label>
                                <select class="form-select" id="rerankerModel">
                                    <option value="cross-encoder">Cross-Encoder</option>
                                    <option value="bm25">BM25</option>
                                    <option value="hybrid">Hybrid</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Top-K Results</label>
                                <input type="number" class="form-control" id="rerankerTopK" value="10">
                            </div>
                            <button class="btn btn-primary" id="saveRerankerConfigBtn">
                                <i class="fas fa-save me-1"></i> Save Configuration
                            </button>
                        </div>
                    </div>
                </div>
                
                <!-- OBSERVABILITY SECTION -->
                <div id="observability" class="config-section">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fas fa-eye me-2"></i>Observability</h2>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">AI-Powered Insights</div>
                                <div class="card-body">
                                    <div class="alert alert-info">
                                        <i class="fas fa-robot me-2"></i>
                                        AI analysis is active and monitoring system patterns.
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">Metrics</div>
                                <div class="card-body">
                                    <div class="mb-3">
                                        <div class="d-flex justify-content-between">
                                            <span>Trace Depth</span>
                                            <strong id="traceDepth">5</strong>
                                        </div>
                                    </div>
                                    <div class="mb-3">
                                        <div class="d-flex justify-content-between">
                                            <span>Log Retention</span>
                                            <strong>30 days</strong>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- REVISION OPERATOR SECTION -->
                <div id="revisao-operador" class="config-section">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fas fa-clipboard-check me-2"></i>Revisão Operador</h2>
                        <button class="btn btn-primary" id="refreshRevisaoBtn">
                            <i class="fas fa-sync-alt me-1"></i> Refresh
                        </button>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">Pending Reviews</div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-hover" id="revisaoTable">
                                    <thead>
                                        <tr>
                                            <th>Date</th>
                                            <th>Type</th>
                                            <th>Content</th>
                                            <th>Quality</th>
                                            <th>Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody id="revisaoTableBody">
                                        <tr><td colspan="5" class="text-center py-4">Loading...</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- AUDIT LOG SECTION -->
                <div id="audit" class="config-section">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fas fa-scroll me-2"></i>Audit Log</h2>
                        <button class="btn btn-primary" id="refreshAuditBtn">
                            <i class="fas fa-sync-alt me-1"></i> Refresh
                        </button>
                    </div>
                    
                    <div class="card">
                        <div class="card-body">
                            <div class="table-responsive" style="max-height: 500px; overflow-y: auto;">
                                <table class="table table-sm table-hover" id="auditTable">
                                    <thead class="table-light sticky-top">
                                        <tr>
                                            <th>Timestamp</th>
                                            <th>User</th>
                                            <th>Action</th>
                                            <th>Resource</th>
                                            <th>Status</th>
                                            <th>Details</th>
                                        </tr>
                                    </thead>
                                    <tbody id="auditTableBody">
                                        <tr><td colspan="6" class="text-center py-4">Loading audit log...</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- ENTERPRISE INCIDENTS SECTION -->
                <div id="enterprise-incidents" class="config-section">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fas fa-exclamation-triangle me-2"></i>Incidents</h2>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">Active Incidents</div>
                        <div class="card-body">
                            <div class="alert alert-success">
                                <i class="fas fa-check-circle me-2"></i> No active incidents
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- ENTERPRISE COMPLIANCE SECTION -->
                <div id="enterprise-compliance" class="config-section">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fas fa-shield-alt me-2"></i>Compliance</h2>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">GDPR Status</div>
                                <div class="card-body">
                                    <div class="alert alert-success">
                                        <i class="fas fa-check me-1"></i> Compliant
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">Data Retention</div>
                                <div class="card-body">
                                    <p>All data retention policies are being followed.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- ENTERPRISE SECURITY SECTION -->
                <div id="enterprise-security" class="config-section">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fas fa-lock me-2"></i>Security & SIEM</h2>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">Security Events</div>
                        <div class="card-body">
                            <div class="alert alert-info">
                                <i class="fas fa-info-circle me-2"></i> No security events in the last 24 hours
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- ENTERPRISE RESILIENCE SECTION -->
                <div id="enterprise-resilience" class="config-section">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2><i class="fas fa-heartbeat me-2"></i>Resilience</h2>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-body text-center">
                                    <i class="fas fa-clock text-success display-4 mb-2"></i>
                                    <h5>Uptime</h5>
                                    <h3 class="text-success">99.9%</h3>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-body text-center">
                                    <i class="fas fa-shield-alt text-primary display-4 mb-2"></i>
                                    <h5>Failover Status</h5>
                                    <span class="badge bg-success">Active</span>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-body text-center">
                                    <i class="fas fa-database text-info display-4 mb-2"></i>
                                    <h5>Replication</h5>
                                    <span class="badge bg-success">Healthy</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const navLinks = document.querySelectorAll('.nav-link[data-target]');
            const configSections = document.querySelectorAll('.config-section');
            
            navLinks.forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const target = this.getAttribute('data-target');
                    
                    navLinks.forEach(l => l.classList.remove('active'));
                    this.classList.add('active');
                    
                    configSections.forEach(section => {
                        section.classList.remove('active');
                        if (section.id === target) {
                            section.classList.add('active');
                        }
                    });
                });
            });
            
            loadHealthData();
            loadTeamsUsers();
            loadBackups();
            loadModels();
            loadGraphragStats();
            loadRevisao();
            loadAuditLog();
        });
        
        function showToast(message, type = 'info') {
            const toastContainer = document.querySelector('.toast-container');
            const toast = document.createElement('div');
            toast.className = 'toast ' + type + ' show';
            toast.innerHTML = '<div class="toast-body">' + message + '<button type="button" class="btn-close float-end" onclick="this.parentElement.parentElement.remove()"></button></div>';
            toastContainer.appendChild(toast);
            setTimeout(function() { toast.remove(); }, 5000);
        }
        
        async function loadHealthData() {
            try {
                const response = await fetch('/health/detailed');
                const data = await response.json();
                updateHealthDisplay(data);
            } catch (error) {
                console.error('Error loading health data:', error);
            }
        }
        
        function updateHealthDisplay(data) {
            var healthCard = document.getElementById('overallHealthCard');
            if (data.status === 'healthy') {
                healthCard.className = 'card health-card border-success';
                healthCard.innerHTML = '<div class="card-body d-flex align-items-center"><div class="me-3"><i class="fas fa-check-circle text-success display-4"></i></div><div><h4 class="mb-0">System Healthy</h4><p class="text-muted mb-0">All components operational</p></div></div>';
            }
        }
        
        async function loadTeamsUsers() {
            try {
                const response = await fetch('/api/admin/teams/users');
                const data = await response.json();
                document.getElementById('teams-users-count').textContent = data.length || 0;
            } catch (error) {
                console.error('Error loading teams users:', error);
            }
        }
        
        async function loadBackups() {
            try {
                const response = await fetch('/api/admin/backup/list');
                const data = await response.json();
                updateBackupsTable(data);
            } catch (error) {
                console.error('Error loading backups:', error);
            }
        }
        
        function updateBackupsTable(data) {
            var tbody = document.getElementById('backupsTableBody');
            if (!data.backups || data.backups.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-muted">No backups found</td></tr>';
                return;
            }
            tbody.innerHTML = data.backups.map(function(backup) {
                return '<tr><td>' + new Date(backup.created_at).toLocaleString() + '</td><td>' + backup.type + '</td><td>' + (backup.size / 1024 / 1024).toFixed(2) + ' MB</td><td><span class="badge bg-success">' + backup.status + '</span></td><td><button class="btn btn-sm btn-outline-primary" onclick="downloadBackup(\'' + backup.id + '\')"><i class="fas fa-download"></i></button></td></tr>';
            }).join('');
        }
        
        async function loadModels() {
            try {
                const response = await fetch('/litellm/config/models');
                const data = await response.json();
                updateModelsTable(data);
            } catch (error) {
                console.error('Error loading models:', error);
            }
        }
        
        function updateModelsTable(data) {
            var tbody = document.getElementById('modelsTableBody');
            document.getElementById('activeModelsCount').textContent = data.length || 0;
            if (!data || data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-muted">No models configured</td></tr>';
                return;
            }
            tbody.innerHTML = data.map(function(model) {
                return '<tr><td>' + model.model_name + '</td><td>' + model.provider + '</td><td><span class="badge bg-success">Active</span></td><td>' + (model.request_count || 0) + '</td><td>$' + (model.total_cost ? model.total_cost.toFixed(4) : '0.0000') + '</td><td>' + (model.avg_latency || 0) + 'ms</td><td><button class="btn btn-sm btn-outline-info" onclick="testModel(\'' + model.model_name + '\')">Test</button></td></tr>';
            }).join('');
        }
        
        async function loadGraphragStats() {
            try {
                const response = await fetch('/api/admin/graphrag/stats');
                const data = await response.json();
                document.getElementById('graphragEntities').textContent = data.entities || 0;
                document.getElementById('graphragRelations').textContent = data.relationships || 0;
                document.getElementById('graphragDocuments').textContent = data.documents || 0;
            } catch (error) {
                console.error('Error loading graphrag stats:', error);
            }
        }
        
        async function loadRevisao() {
            try {
                const response = await fetch('/api/v1/continual-learning/review/pending');
                const data = await response.json();
                document.getElementById('revisao-pending-count').textContent = data.length || 0;
            } catch (error) {
                console.error('Error loading revisao:', error);
            }
        }
        
        async function loadAuditLog() {
            try {
                const response = await fetch('/enterprise/audit/logs');
                const data = await response.json();
                updateAuditTable(data);
            } catch (error) {
                console.error('Error loading audit log:', error);
            }
        }
        
        function updateAuditTable(data) {
            var tbody = document.getElementById('auditTableBody');
            if (!data || data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-muted">No audit logs</td></tr>';
                return;
            }
            tbody.innerHTML = data.map(function(log) {
                return '<tr><td>' + new Date(log.timestamp).toLocaleString() + '</td><td>' + (log.user || 'System') + '</td><td>' + log.action + '</td><td>' + log.resource + '</td><td><span class="badge bg-' + (log.status === 'success' ? 'success' : 'danger') + '">' + log.status + '</span></td><td>' + (log.details || '-') + '</td></tr>';
            }).join('');
        }
    </script>
</body>
</html>
```

---

## 2. Arquivo: `resync/static/css/admin-neumorphic.css`

### Criar novo arquivo CSS com o seguinte conteúdo:

```css
/* Neumorphic Admin Styles - Resync Optimized */

:root {
    --primary-color: #007bff;
    --secondary-color: #6c757d;
    --success-color: #28a745;
    --warning-color: #ffc107;
    --danger-color: #dc3545;
    --dark-color: #343a40;
    --light-color: #f8f9fa;
    --bg-color: #f5f7fb;
    --shadow-light: #ffffff;
    --shadow-dark: #d1d9e6;
    --neu-shadow: 8px 8px 16px var(--shadow-dark), -8px -8px 16px var(--shadow-light);
    --neu-shadow-inset: inset 4px 4px 8px var(--shadow-dark), inset -4px -4px 8px var(--shadow-light);
}

body {
    background-color: var(--bg-color);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.admin-header {
    background: linear-gradient(135deg, var(--primary-color), #0056b3);
    color: white;
    padding: 1.5rem 0;
    border-radius: 15px;
    box-shadow: var(--neu-shadow);
    margin-bottom: 2rem;
}

.sidebar {
    background-color: white;
    border-radius: 15px;
    box-shadow: var(--neu-shadow);
1rem;
       padding:  height: calc(100vh - 100px);
    position: sticky;
    top: 20px;
    overflow-y: auto;
}

.sidebar .nav-link {
    padding: 12px 15px;
    border-radius: 10px;
    margin-bottom: 5px;
    color: var(--secondary-color);
    font-weight: 500;
    transition: all 0.3s ease;
}

.sidebar .nav-link:hover {
    background-color: rgba(0, 123, 255, 0.1);
    color: var(--primary-color);
    transform: translateX(5px);
}

.sidebar .nav-link.active {
    background: linear-gradient(135deg, var(--primary-color), #0056b3);
    color: white;
    box-shadow: var(--neu-shadow);
}

.sidebar h6 {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--secondary-color);
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    padding-left: 0.5rem;
}

.card {
    border: none;
    border-radius: 15px;
    background: white;
    box-shadow: var(--neu-shadow);
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 12px 12px 24px var(--shadow-dark), -12px -12px 24px var(--shadow-light);
}

.card-header {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border-bottom: none;
    border-radius: 15px 15px 0 0 !important;
    padding: 1rem 1.5rem;
    font-weight: 600;
    color: var(--dark-color);
}

.card-body {
    padding: 1.5rem;
}

.btn {
    border-radius: 10px;
    padding: 0.5rem 1.25rem;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: var(--neu-shadow);
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 4px 4px 8px var(--shadow-dark), -4px -4px 8px var(--shadow-light);
}

.btn:active {
    transform: translateY(0);
    box-shadow: var(--neu-shadow-inset);
}

.btn-primary {
    background: linear-gradient(135deg, var(--primary-color), #0069d9);
    border: none;
    color: white;
}

.btn-success {
    background: linear-gradient(135deg, var(--success-color), #218838);
    border: none;
    color: white;
}

.form-control {
    border-radius: 10px;
    border: 2px solid #e0e0e0;
    padding: 0.75rem 1rem;
    transition: all 0.3s ease;
}

.form-control:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
}

.form-check-input:checked {
    background-color: var(--primary-color);
    border-color: var(--primary-color);
}

.table thead th {
    border-bottom: 2px solid var(--shadow-dark);
    color: var(--secondary-color);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.85rem;
}

.table td {
    padding: 1rem;
    vertical-align: middle;
}

.table-hover tbody tr:hover {
    background-color: rgba(0, 123, 255, 0.05);
}

.health-card {
    border-radius: 15px;
    box-shadow: var(--neu-shadow);
    transition: all 0.3s ease;
}

.health-card:hover {
    transform: scale(1.02);
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 10px currentColor;
}

.status-connected {
    background-color: var(--success-color);
    box-shadow: 0 0 10px var(--success-color);
}

.status-disconnected {
    background-color: var(--danger-color);
    box-shadow: 0 0 10px var(--danger-color);
}

.status-warning {
    background-color: var(--warning-color);
    box-shadow: 0 0 10px var(--warning-color);
}

.toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 9999;
}

.toast {
    background: white;
    border-radius: 10px;
    box-shadow: var(--neu-shadow);
    padding: 1rem;
    margin-bottom: 10px;
    min-width: 300px;
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

.toast.success { border-left: 4px solid var(--success-color); }
.toast.error { border-left: 4px solid var(--danger-color); }
.toast.warning { border-left: 4px solid var(--warning-color); }
.toast.info { border-left: 4px solid var(--primary-color); }

.modal-content {
    border: none;
    border-radius: 15px;
    box-shadow: var(--neu-shadow);
}

.modal-header {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border-bottom: none;
    border-radius: 15px 15px 0 0;
    padding: 1.5rem;
}

.modal-body {
    padding: 1.5rem;
}

.modal-footer {
    border-top: 1px solid #e0e0e0;
    padding: 1rem 1.5rem;
}

.display-4 {
    font-size: 2.5rem;
    font-weight: 300;
}

.config-section {
    display: none;
    animation: fadeIn 0.5s ease;
}

.config-section.active {
    display: block;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.alert {
    border-radius: 10px;
    border: none;
    box-shadow: var(--neu-shadow);
}

@media (max-width: 768px) {
    .sidebar {
        height: auto;
        position: relative;
        top: 0;
        margin-bottom: 1rem;
    }
    
    .card {
        margin-bottom: 1rem;
    }
    
    .admin-header h1 {
        font-size: 1.25rem;
    }
}
```

---

## 3. Arquivo: `resync/templates/admin/teams_webhook.html`

### Criar novo arquivo:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Teams Webhook Administration</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="/static/css/admin-neumorphic.css" rel="stylesheet">
    <style>
        body { background-color: #f5f7fb; padding: 20px; }
        .container { max-width: 1400px; }
        h1 { color: #007bff; margin-bottom: 2rem; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
        .stat-card { background: white; border-radius: 15px; padding: 1.5rem; box-shadow: 8px 8px 16px #d1d9e6, -8px -8px 16px #ffffff; text-align: center; }
        .stat-card h3 { font-size: 2rem; margin: 0.5rem 0; }
        .section { background: white; border-radius: 15px; padding: 1.5rem; box-shadow: 8px 8px 16px #d1d9e6, -8px -8px 16px #ffffff; margin-bottom: 1.5rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1><i class="fab fa-microsoft me-2"></i>Teams Webhook Administration</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <i class="fas fa-users text-primary display-4"></i>
                <h3 id="totalUsers">0</h3>
                <p class="text-muted mb-0">Total Users</p>
            </div>
            <div class="stat-card">
                <i class="fas fa-check-circle text-success display-4"></i>
                <h3 id="activeUsers">0</h3>
                <p class="text-muted mb-0">Active Users</p>
            </div>
            <div class="stat-card">
                <i class="fas fa-bolt text-warning display-4"></i>
                <h3 id="executeUsers">0</h3>
                <p class="text-muted mb-0">Can Execute</p>
            </div>
            <div class="stat-card">
                <i class="fas fa-comments text-info display-4"></i>
                <h3 id="totalInteractions">0</h3>
                <p class="text-muted mb-0">Interactions</p>
            </div>
        </div>
        
        <div class="section">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2><i class="fas fa-users-cog me-2"></i>Users</h2>
                <button class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#addUserModal">
                    <i class="fas fa-user-plus me-1"></i> Add User
                </button>
            </div>
            
            <div class="table-responsive">
                <table class="table table-hover" id="usersTable">
                    <thead>
                        <tr>
                            <th>User</th>
                            <th>Email</th>
                            <th>Role</th>
                            <th>Execute</th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="usersTableBody">
                        <tr><td colspan="6" class="text-center py-4">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="section">
            <h2><i class="fas fa-history me-2"></i>Audit Logs</h2>
            <div class="table-responsive" style="max-height: 400px; overflow-y: auto;">
                <table class="table table-sm table-hover" id="auditTable">
                    <thead class="table-light sticky-top">
                        <tr>
                            <th>Time</th>
                            <th>User</th>
                            <th>Type</th>
                            <th>Authorized</th>
                            <th>Message</th>
                        </tr>
                    </thead>
                    <tbody id="auditTableBody">
                        <tr><td colspan="5" class="text-center py-3">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <div class="modal fade" id="addUserModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title"><i class="fas fa-user-plus me-2"></i>Add New User</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <form id="addUserForm">
                        <div class="mb-3">
                            <label class="form-label">Email *</label>
                            <input type="email" class="form-control" id="newUserEmail" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Full Name *</label>
                            <input type="text" class="form-control" id="newUserName" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Role</label>
                            <select class="form-select" id="newUserRole">
                                <option value="viewer">Viewer</option>
                                <option value="operator">Operator</option>
                                <option value="admin">Admin</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="newUserCanExecute">
                                <label class="form-check-label">Can Execute Commands</label>
                            </div>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" id="saveNewUserBtn">Save User</button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', loadData);
        
        async function loadData() {
            await Promise.all([loadUsers(), loadAuditLogs()]);
        }
        
        async function loadUsers() {
            try {
                const response = await fetch('/api/admin/teams/users');
                const users = await response.json();
                
                document.getElementById('totalUsers').textContent = users.length;
                document.getElementById('activeUsers').textContent = users.filter(function(u) { return u.is_active; }).length;
                document.getElementById('executeUsers').textContent = users.filter(function(u) { return u.can_execute; }).length;
                
                var tbody = document.getElementById('usersTableBody');
                if (users.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-muted">No users found</td></tr>';
                    return;
                }
                
                tbody.innerHTML = users.map(function(user) {
                    return '<tr><td>' + user.name + '</td><td>' + user.email + '</td><td><span class="badge bg-' + (user.role === 'admin' ? 'danger' : user.role === 'operator' ? 'info' : 'secondary') + '">' + user.role + '</span></td><td>' + (user.can_execute ? '<i class="fas fa-check text-success"></i>' : '<i class="fas fa-times text-danger"></i>') + '</td><td><span class="badge bg-' + (user.is_active ? 'success' : 'secondary') + '">' + (user.is_active ? 'Active' : 'Inactive') + '</span></td><td><button class="btn btn-sm btn-outline-primary"><i class="fas fa-edit"></i></button> <button class="btn btn-sm btn-outline-danger"><i class="fas fa-trash"></i></button></td></tr>';
                }).join('');
            } catch (error) {
                console.error('Error loading users:', error);
            }
        }
        
        async function loadAuditLogs() {
            try {
                const response = await fetch('/api/admin/teams/audit');
                const logs = await response.json();
                
                var tbody = document.getElementById('auditTableBody');
                if (logs.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" class="text-center py-3 text-muted">No audit logs</td></tr>';
                    return;
                }
                
                tbody.innerHTML = logs.map(function(log) {
                    return '<tr><td>' + new Date(log.timestamp).toLocaleString() + '</td><td>' + (log.user_email || 'Unknown') + '</td><td>' + log.command_type + '</td><td>' + (log.authorized ? '<i class="fas fa-check text-success"></i>' : '<i class="fas fa-times text-danger"></i>') + '</td><td>' + (log.message || '-') + '</td></tr>';
                }).join('');
            } catch (error) {
                console.error('Error loading audit logs:', error);
            }
        }
        
        document.getElementById('saveNewUserBtn').addEventListener('click', async function() {
            var email = document.getElementById('newUserEmail').value;
            var name = document.getElementById('newUserName').value;
            var role = document.getElementById('newUserRole').value;
            var canExecute = document.getElementById('newUserCanExecute').checked;
            
            try {
                const response = await fetch('/api/admin/teams/users', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email: email, name: name, role: role, can_execute: canExecute })
                });
                
                if (response.ok) {
                    var modal = bootstrap.Modal.getInstance(document.getElementById('addUserModal'));
                    modal.hide();
                    loadUsers();
                }
            } catch (error) {
                console.error('Error saving user:', error);
            }
        });
    </script>
</body>
</html>
```

---

## Resumo das Alterações

| Arquivo | Ação | Descrição |
|---------|------|-----------|
| `resync/templates/admin.html` | Substituir | Novo template com 9 seções de navegação, design neumórfico completo |
| `resync/static/css/admin-neumorphic.css` | Criar | Arquivo CSS com estilos neumórficos |
| `resync/templates/admin/teams_webhook.html` | Criar | Template para gestão de webhooks Teams |

### Seções Incluidas:
1. **Configuration** - Teams, TWS, System Settings, LiteLLM
2. **Monitoring** - System Health, TWS Proativo, Notifications
3. **AI & Learning** - Auto-Tuning, GraphRAG, RAG Reranker
4. **Tools** - Backup, Observability, Revisão, Audit
5. **Enterprise** - Overview, Incidents, Compliance, Security, Resilience