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
        
        // Check if user is authenticated before connecting
        // Check for admin_logged_in flag (set by /admin login) or access_token
        const isLoggedIn = sessionStorage.getItem('admin_logged_in') === '1' || 
                          localStorage.getItem('admin_logged_in') === '1' ||
                          sessionStorage.getItem('access_token') || 
                          localStorage.getItem('access_token');
        if (!isLoggedIn) {
            this.updateStatus('disconnected', 'Please login to use chat');
            // Save return URL so admin login can redirect back to chat
            sessionStorage.setItem('return_url', window.location.pathname);
            // Redirect to admin page which has login modal
            setTimeout(() => {
                window.location.href = '/admin';
            }, 1500);
            return;
        }
        
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

    getAuthToken() {
        // Try to get token from localStorage first (set by admin login)
        let token = localStorage.getItem('access_token') || sessionStorage.getItem('access_token');
        
        // If no token, check for admin_logged_in flag which implies cookie auth
        if (!token && (localStorage.getItem('admin_logged_in') === '1' || sessionStorage.getItem('admin_logged_in') === '1')) {
            console.log('[Chat] Using cookie-based auth (admin_logged_in flag present)');
            return null; // Rely on cookie
        }
        
        return token;
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        
        // Try to get token from cookie or sessionStorage
        const token = this.getAuthToken();
        const wsUrl = token 
            ? `${protocol}//${window.location.host}/ws/tws-general?token=${token}`
            : `${protocol}//${window.location.host}/ws/tws-general`;

        console.log('[Chat] WebSocket URL:', wsUrl.replace(token, '***'));

        this.updateStatus('disconnected', 'Connecting...');

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('[Chat] WebSocket connected successfully');
                this.reconnectAttempts = 0;
                this.updateStatus('connected', 'Connected');
            };

            this.ws.onmessage = (event) => {
                console.log('[Chat] Raw message received:', event.data.substring ? event.data.substring(0, 500) : event.data);
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (e) {
                    console.error('[Chat] Error parsing message:', e);
                }
            };

            this.ws.onerror = (error) => {
                console.error('[Chat] WebSocket error:', error);
                // Status update handled by onclose usually, but we can set here too
            };

            this.ws.onclose = (event) => {
                console.log('[Chat] WebSocket closed:', { code: event.code, reason: event.reason });
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
        console.log('[Chat] Received message:', JSON.stringify(data));
        
        // Remove typing indicator
        this.hideTypingIndicator();

        // Update trace ID if present - check both top level and metadata
        const correlationId = data.correlation_id || (data.metadata && data.metadata.correlation_id);
        if (correlationId && this.traceIdDisplay) {
            this.traceId = correlationId;
            // Use textContent for trace ID to prevent XSS
            this.traceIdDisplay.textContent = `Trace: ${correlationId.substring(0, 8)}...`;
        }

        // Handle system messages
        if (data.type === 'system') {
            this.addMessage('system', data.message);
            return;
        }

        // Display assistant response - server sends "message" field, not "response"
        console.log('[Chat] Checking message display:', { 
            hasMessage: !!data.message, 
            isFinal: data.is_final,
            type: data.type,
            sender: data.sender,
            messageLength: data.message ? data.message.length : 0
        });
        
        if (data.message && data.is_final) {
            // Get metadata from the response
            const metadata = data.metadata || {};
            console.log('[Chat] Displaying assistant message:', data.message.substring(0, 100));
            this.addMessage('assistant', data.message, {
                mode: metadata.routing_mode || data.routing_mode,
                intent: metadata.intent,
                confidence: metadata.confidence,
                agent: data.agent_id
            });
            return;
        }

        // Handle streaming/in-progress messages
        if (data.message && !data.is_final) {
            console.log('[Chat] In-progress message received');
            return;
        }

        // Handle direct text messages (some WS implementations might send raw text)
        if (data.type === 'message' && data.content) {
            console.log('[Chat] Displaying raw text message');
            this.addMessage('assistant', data.content);
        }

        // Handle errors
        if (data.error) {
            console.log('[Chat] Error received:', data.error);
            this.addMessage('system', `Error: ${data.error}`);
        }
        
        // Log unhandled message types
        console.log('[Chat] Unhandled message type:', data);
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
