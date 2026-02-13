document.addEventListener('DOMContentLoaded', function() {
    // --- DOM Element Selectors ---
    const workstationsTotalEl = document.getElementById('workstations-total');
    const jobsAbendEl = document.getElementById('jobs-abend');
    const jobsSuccEl = document.getElementById('jobs-succ');
    const twsStatusTextEl = document.getElementById('tws-status-text');
    const twsConnectionStatusEl = document.getElementById('tws-connection-status');

    // TWS Instance selector (for multi-server support)
    const twsInstanceSelectEl = document.getElementById('tws-instance-select');
    const twsSelectorContainerEl = document.getElementById('tws-selector-container');
    
    // Chat elements
    const chatMessagesEl = document.getElementById('chat-messages');
    const chatInputEl = document.getElementById('chat-input');
    const sendButtonEl = document.getElementById('send-button');
    const wsStatusTextEl = document.getElementById('ws-status-text');
    const websocketStatusEl = document.getElementById('websocket-status');

    // RAG Upload Elements
    const fileInputEl = document.getElementById('file-input');
    const uploadButtonEl = document.getElementById('upload-button');
    const uploadStatusEl = document.getElementById('upload-status');

    let websocket = null;

    // --- UI Update Functions ---
    const updateTWSConnectionStatus = function(isOnline) {
        if (isOnline) {
            twsConnectionStatusEl.classList.remove('offline');
            twsConnectionStatusEl.classList.add('online');
            twsStatusTextEl.textContent = 'Disponível';
        } else {
            twsConnectionStatusEl.classList.remove('online');
            twsConnectionStatusEl.classList.add('offline');
            twsStatusTextEl.textContent = 'Indisponível';
        }
    };

    const updateWebSocketStatus = function(isConnected) {
        if (isConnected) {
            websocketStatusEl.classList.remove('offline');
            websocketStatusEl.classList.add('online');
            wsStatusTextEl.textContent = 'Conectado';
            chatInputEl.disabled = false;
            sendButtonEl.disabled = false;
        } else {
            websocketStatusEl.classList.remove('online');
            websocketStatusEl.classList.add('offline');
            wsStatusTextEl.textContent = 'Desconectado';
            chatInputEl.disabled = true;
            sendButtonEl.disabled = true;
        }
    };

    const addChatMessage = function(sender, message, type = 'message') {
        const messageEl = document.createElement('div');
        messageEl.classList.add('message', sender, type);
        messageEl.textContent = message;
        chatMessagesEl.appendChild(messageEl);
        chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight; // Auto-scroll
    };

    // --- Data Fetching ---
    const fetchSystemStatus = async function() {
        try {
            const response = await fetch('/api/status');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();

            // Update dashboard metrics
            workstationsTotalEl.textContent = data.workstations.length;
            jobsAbendEl.textContent = data.jobs.filter(function(j) { return j.status === 'ABEND'; }).length;
            jobsSuccEl.textContent = data.jobs.filter(function(j) { return j.status === 'SUCC'; }).length;
            updateTWSConnectionStatus(true);
        } catch (error) {
            console.error('Failed to fetch system status:', error);
            updateTWSConnectionStatus(false);
            // Reset metrics on failure
            workstationsTotalEl.textContent = '--';
            jobsAbendEl.textContent = '--';
            jobsSuccEl.textContent = '--';
        }
    };

    // fetchAgents removed - routing is now automatic via UnifiedAgent
    // No manual agent selection needed

    // --- Fetch TWS Instances (v5.2.3.25) ---
    const fetchTWSInstances = async function() {
        if (!twsInstanceSelectEl) return;
        
        try {
            const response = await fetch('/api/v1/admin/tws-instances');
            if (!response.ok) throw new Error('Failed to fetch TWS instances');
            const data = await response.json();
            const instances = data.instances || [];

            // Only show selector when there are multiple instances
            if (instances.length <= 1) {
                if (twsSelectorContainerEl) {
                    twsSelectorContainerEl.style.display = 'none';
                }
                // If only one instance, auto-select it
                if (instances.length === 1) {
                    twsInstanceSelectEl.innerHTML = `<option value="${instances[0].id}">${instances[0].display_name || instances[0].name}</option>`;
                    twsInstanceSelectEl.value = instances[0].id;
                }
                return;
            }
            
            // Show selector for multiple instances
            if (twsSelectorContainerEl) {
                twsSelectorContainerEl.style.display = 'block';
            }

            // Sort by display_name
            instances.sort((a, b) => (a.display_name || a.name).localeCompare(b.display_name || b.name));

            twsInstanceSelectEl.innerHTML = '<option value="">Selecione o servidor TWS</option>';
            instances.forEach(function(inst) {
                const option = document.createElement('option');
                option.value = inst.id;
                option.textContent = inst.display_name || inst.name;
                option.style.borderLeft = '3px solid ' + (inst.color || '#3b82f6');
                
                // Add status indicator
                const statusIcon = inst.status === 'connected' ? '🟢' : 
                                   inst.status === 'error' ? '🔴' : '⚪';
                option.textContent = statusIcon + ' ' + option.textContent;
                
                // Add environment tag
                if (inst.environment === 'production') {
                    option.textContent += ' [PROD]';
                } else if (inst.environment === 'staging') {
                    option.textContent += ' [STG]';
                } else if (inst.environment === 'dr') {
                    option.textContent += ' [DR]';
                }
                
                twsInstanceSelectEl.appendChild(option);
            });

            // Select first connected instance by default
            const connectedInstance = instances.find(i => i.status === 'connected');
            if (connectedInstance) {
                twsInstanceSelectEl.value = connectedInstance.id;
            }
        } catch (error) {
            console.error('Failed to fetch TWS instances:', error);
            if (twsSelectorContainerEl) {
                twsSelectorContainerEl.style.display = 'none';
            }
        }
    };

    // --- RAG File Upload ---
    const uploadFile = async function() {
        const file = fileInputEl.files[0];
        if (!file) {
            uploadStatusEl.textContent = 'Por favor, selecione um arquivo.';
            uploadStatusEl.className = 'upload-status error';
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        uploadStatusEl.textContent = 'Enviando arquivo...';
        uploadStatusEl.className = 'upload-status info';

        try {
            const response = await fetch('/api/rag/upload', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            if (response.ok) {
                uploadStatusEl.textContent = `Arquivo '${result.filename}' enviado com sucesso!`;
                uploadStatusEl.className = 'upload-status success';
                fileInputEl.value = ''; // Clear the input
            } else {
                throw new Error(result.detail || 'Falha no envio do arquivo.');
            }
        } catch (error) {
            console.error('File upload error:', error);
            uploadStatusEl.textContent = `Erro: ${error.message}`;
            uploadStatusEl.className = 'upload-status error';
        }
    };

    // --- WebSocket Management ---
    const connectWebSocket = function() {
        if (websocket) {
            websocket.close();
        }

        // Use unified agent endpoint - routing is automatic
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/ws/unified`;
        websocket = new WebSocket(wsUrl);

        websocket.onopen = function() {
            console.log('WebSocket connection established (unified agent).');
            updateWebSocketStatus(true);
        };

        websocket.onmessage = function(event) {
            const data = JSON.parse(event.data);
            console.log('WebSocket message received:', data);

            if (data.type === 'stream') {
                const lastMessage = chatMessagesEl.querySelector('.message.agent:last-child');
                if (lastMessage && !lastMessage.dataset.final) {
                    lastMessage.textContent += data.message;
                } else {
                     addChatMessage(data.sender, data.message);
                }
            } else if (data.type === 'message' && data.is_final) {
                const lastMessage = chatMessagesEl.querySelector('.message.agent:last-child');
                 if (lastMessage && !lastMessage.dataset.final) {
                    lastMessage.textContent = data.message;
                    lastMessage.dataset.final = true;
                } else {
                    addChatMessage(data.sender, data.message);
                }
            } else {
                 addChatMessage(data.sender, data.message, data.type);
            }
        };

        websocket.onclose = function() {
            console.log('WebSocket connection closed.');
            updateWebSocketStatus(false);
            websocket = null;
        };

        websocket.onerror = function(error) {
            console.error('WebSocket error:', error);
            addChatMessage('system', 'Erro na conexão com o WebSocket.', 'error');
            updateWebSocketStatus(false);
        };
    };

    const sendMessage = function() {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            const trimmed = chatInputEl.value.trim();
            // Don't send empty messages
            if (!trimmed) {
                return;
            }
            // Build a structured payload - routing is automatic via UnifiedAgent
            // v5.2.3.25: Include tws_instance_id for multi-instance support
            const payload = {
                type: 'chat_message',
                content: trimmed,
                tws_instance_id: twsInstanceSelectEl?.value || null
            };
            try {
                websocket.send(JSON.stringify(payload));
            } catch (err) {
                console.error('Failed to send WebSocket payload:', err);
                addChatMessage('system', 'Falha ao enviar mensagem.', 'error');
                return;
            }
            // Reflect the user's message in the chat window immediately
            addChatMessage('user', trimmed);
            chatInputEl.value = '';
        }
    };

    // --- Event Listeners ---
    sendButtonEl.addEventListener('click', sendMessage);
    chatInputEl.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
    uploadButtonEl.addEventListener('click', uploadFile);

    // --- Initial Load ---
    const initializeDashboard = function() {
        fetchSystemStatus();
        fetchTWSInstances();  // v5.2.3.25: Load TWS instances
        connectWebSocket();   // Connect to unified agent automatically
        setInterval(fetchSystemStatus, 30000); // Refresh status every 30 seconds
        setInterval(fetchTWSInstances, 60000); // Refresh TWS instances every 60 seconds
    };

    initializeDashboard();
});
