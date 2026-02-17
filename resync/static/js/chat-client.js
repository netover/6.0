class ChatClient {
    constructor() {
        this.ws = null;
        this.traceId = null;
        this.messages = [];
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.baseTitle = document.title;

        this.init();
    }

    init() {
        this.setupElements();
        this.setupEventListeners();
        this.connect();
    }

    setupElements() {
        this.messagesContainer = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.chatForm = document.getElementById('chatForm');
        this.statusIndicator = document.querySelector('.status-indicator');
        this.statusText = document.getElementById('statusText');
        this.traceIdDisplay = document.getElementById('traceId');
    }

    setupEventListeners() {
        this.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });

        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = this.messageInput.scrollHeight + 'px';
        });

        // Send on Ctrl+Enter
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                this.sendMessage();
            }
        });
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/chat`;

        this.updateStatus('disconnected', 'Connecting...');

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.updateStatus('connected', 'Connected');
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (e) {
                    console.error('Error parsing message:', e);
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                // Status update handled by onclose usually, but we can set here too
            };

            this.ws.onclose = () => {
                console.log('WebSocket closed');
                this.updateStatus('disconnected', 'Disconnected');
                this.attemptReconnect();
            };
        } catch (e) {
            console.error('WebSocket connection failed:', e);
            this.attemptReconnect();
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);

            this.updateStatus('disconnected', `Reconnecting in ${delay / 1000}s...`);

            setTimeout(() => {
                console.log(`Reconnect attempt ${this.reconnectAttempts}`);
                this.connect();
            }, delay);
        } else {
            this.updateStatus('disconnected', 'Connection failed');
        }
    }

    updateStatus(status, text) {
        if (this.statusIndicator && this.statusText) {
            this.statusIndicator.className = `status-indicator ${status}`;
            this.statusText.textContent = text;
        }
    }

    sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.addMessage('system', 'Connection lost. Trying to reconnect...');
            this.connect();
            return;
        }

        // Display user message immediately
        this.addMessage('user', message);

        // Send to server
        this.ws.send(JSON.stringify({
            message: message,
            agent_id: 'default'
        }));

        // Clear input
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';

        // Show typing indicator
        this.showTypingIndicator();
    }

    handleMessage(data) {
        // Remove typing indicator
        this.hideTypingIndicator();

        // Update trace ID if present
        if (data.trace_id && this.traceIdDisplay) {
            this.traceId = data.trace_id;
            // Use textContent for trace ID to prevent XSS
            this.traceIdDisplay.textContent = `Trace: ${data.trace_id.substring(0, 8)}...`;
        }

        // Display assistant response
        if (data.response) {
            this.addMessage('assistant', data.response, {
                mode: data.routing_mode || data.mode, // Handle different formats
                intent: data.intent,
                confidence: data.confidence,
                agent: data.agent_id || data.handler
            });
        }

        // Handle direct text messages (some WS implementations might send raw text)
        if (data.type === 'message' && data.content) {
            this.addMessage('assistant', data.content);
        }

        // Handle errors
        if (data.error) {
            this.addMessage('system', `Error: ${data.error}`);
        }
    }

    addMessage(role, content, metadata = {}) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${role}`;

        const avatar = role === 'user' ? '👤' : role === 'assistant' ? '🤖' : '⚠️';

        // Build metadata HTML with proper escaping
        let metaHtml = '';
        if (role === 'assistant' && (metadata.mode || metadata.intent)) {
            const confidenceStr = metadata.confidence ? ` | ${(metadata.confidence * 100).toFixed(0)}%` : '';
            // Escape all metadata values to prevent XSS
            const modeStr = metadata.mode ? escapeHtml(metadata.mode) : '';
            const intentStr = metadata.intent ? escapeHtml(metadata.intent) : '';

            metaHtml = `
                <div class="message-meta">
                    ${modeStr ? `Mode: ${modeStr}` : ''} 
                    ${intentStr ? `| Intent: ${intentStr}` : ''}
                    ${confidenceStr}
                </div>
            `;
        }

        // Use textContent for the avatar and properly escape content
        // The formatMessage function already escapes HTML, so it's safe to use innerHTML
        messageEl.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                ${this.formatMessage(content)}
                ${metaHtml}
            </div>
        `;

        this.messagesContainer.appendChild(messageEl);
        this.scrollToBottom();
    }

    formatMessage(content) {
        if (!content) return '';

        // First escape HTML to prevent XSS, then apply markdown-like formatting
        // This ensures user content is safe before any formatting is applied
        return escapeHtml(content)
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    showTypingIndicator() {
        // Only show if not already showing
        if (document.getElementById('typingIndicator')) return;

        const indicator = document.createElement('div');
        indicator.id = 'typingIndicator';
        indicator.className = 'message assistant';
        // Static HTML, no user input - safe to use innerHTML
        indicator.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        this.messagesContainer.appendChild(indicator);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    window.chatClient = new ChatClient();
});
