/*
  FeedbackManager (Chat UI)
  - Additive: does not change existing chat-client.js
  - Uses MutationObserver to inject feedback UI for assistant messages.
  - Captures query_text + response_text (bounded) for higher quality feedback.

  Payload -> POST /api/v1/feedback/submit
*/

(() => {
  'use strict';

  class FeedbackManager {
    constructor(chatClient) {
      this.chatClient = chatClient || null;
      this.apiUrl = '/api/v1/feedback/submit';
      this.maxTextLen = 20000;
      this.assistantIndex = 0;

      this.init();
    }

    init() {
      this.messagesContainer = document.getElementById('chatMessages');
      if (!this.messagesContainer) return;

      this.ensureModalScaffold();
      this.setupMessageObserver();

      // Inject for already-rendered messages
      this.bootstrapExisting();
    }

    bootstrapExisting() {
      const assistantMessages = this.messagesContainer.querySelectorAll('.message.assistant');
      assistantMessages.forEach((el) => this.ensureFeedbackUI(el));
    }

    setupMessageObserver() {
      const observer = new MutationObserver((mutations) => {
        for (const m of mutations) {
          for (const node of m.addedNodes) {
            if (!(node instanceof HTMLElement)) continue;

            // Message might be directly the .message, or contain messages inside
            if (node.classList && node.classList.contains('message') && node.classList.contains('assistant')) {
              this.ensureFeedbackUI(node);
              continue;
            }

            const inner = node.querySelectorAll ? node.querySelectorAll('.message.assistant') : [];
            inner.forEach((el) => this.ensureFeedbackUI(el));
          }
        }
      });

      observer.observe(this.messagesContainer, { childList: true, subtree: true });
      this.observer = observer;
    }

    ensureFeedbackUI(messageEl) {
      if (!messageEl || !(messageEl instanceof HTMLElement)) return;
      if (messageEl.dataset.chatfbAttached === '1') return;

      // Assign a stable message index for correlation.
      // If the element already has one, keep it.
      if (!messageEl.dataset.chatfbMessageIndex) {
        messageEl.dataset.chatfbMessageIndex = String(this.assistantIndex++);
      }

      const contentEl = messageEl.querySelector('.message-content');
      if (!contentEl) return;

      // Capture context
      const responseText = this.extractMessageText(messageEl);
      const queryText = this.extractPreviousUserText(messageEl);
      messageEl.dataset.chatfbQueryText = this.boundText(queryText);
      messageEl.dataset.chatfbResponseText = this.boundText(responseText);

      // Build bar
      const bar = document.createElement('div');
      bar.className = 'chatfb-bar';

      const upBtn = this.buildIconButton('👍', 'Gostei', 'chatfb-btn chatfb-btn--icon chatfb-btn--up');
      const downBtn = this.buildIconButton('👎', 'Não gostei', 'chatfb-btn chatfb-btn--icon chatfb-btn--down');
      const commentBtn = this.buildTextButton('💬', 'Info', 'chatfb-btn');
      const status = document.createElement('div');
      status.className = 'chatfb-status';
      status.textContent = '—';
      status.style.display = 'none';

      upBtn.addEventListener('click', () => this.onThumb(messageEl, +5, upBtn, downBtn, status));
      downBtn.addEventListener('click', () => this.onThumb(messageEl, +1, downBtn, upBtn, status));
      commentBtn.addEventListener('click', () => this.openCommentModal(messageEl, status));

      bar.appendChild(upBtn);
      bar.appendChild(downBtn);
      bar.appendChild(commentBtn);
      bar.appendChild(status);

      // Append after existing content + meta
      contentEl.appendChild(bar);

      messageEl.dataset.chatfbAttached = '1';
    }

    buildIconButton(icon, ariaLabel, className) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = className;
      btn.setAttribute('aria-label', ariaLabel);
      btn.textContent = icon;
      return btn;
    }

    buildTextButton(icon, label, className) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = className;
      btn.setAttribute('aria-label', label);
      btn.innerHTML = `<span aria-hidden="true">${icon}</span><span class="chatfb-label">${label}</span>`;
      return btn;
    }

    extractMessageText(messageEl) {
      const contentEl = messageEl.querySelector('.message-content');
      if (!contentEl) return '';

      // Clone to remove meta and feedback bar before extracting text
      const clone = contentEl.cloneNode(true);
      const meta = clone.querySelector('.message-meta');
      if (meta) meta.remove();
      const fb = clone.querySelector('.chatfb-bar');
      if (fb) fb.remove();

      // The first children include formatted text with <br>. textContent preserves line breaks poorly,
      // but innerText keeps visible newlines.
      const text = (clone.innerText || clone.textContent || '').trim();
      return text;
    }

    extractPreviousUserText(assistantEl) {
      // Find the closest previous sibling with .message.user
      let prev = assistantEl.previousElementSibling;
      while (prev) {
        if (prev.classList.contains('message') && prev.classList.contains('user')) {
          return this.extractMessageText(prev);
        }
        prev = prev.previousElementSibling;
      }
      return '';
    }

    boundText(text) {
      const t = (text || '').trim();
      if (!t) return '';
      if (t.length <= this.maxTextLen) return t;
      return t.slice(0, this.maxTextLen);
    }

    getTraceId() {
      // best-effort
      const fromClient = this.chatClient && this.chatClient.traceId ? String(this.chatClient.traceId) : '';
      const footer = document.getElementById('traceId');
      const fromFooter = footer ? (footer.textContent || '') : '';

      // Footer shows "Trace: abcd..."; try to parse a longer one first
      if (fromClient) return fromClient;

      const m = fromFooter.match(/Trace:\s*([a-zA-Z0-9\-_]+)/);
      if (m && m[1]) return m[1];
      return 'unknown';
    }

    async onThumb(messageEl, rating, primaryBtn, secondaryBtn, statusEl) {
      if (primaryBtn.disabled) return;

      // Visual state
      primaryBtn.classList.add('is-selected');
      secondaryBtn.classList.remove('is-selected');
      primaryBtn.disabled = true;
      secondaryBtn.disabled = true;

      statusEl.style.display = 'inline-flex';
      statusEl.textContent = 'Enviando…';

      const ok = await this.submitFeedback({
        messageEl,
        rating,
        feedback_type: rating >= 4 ? 'helpfulness' : 'accuracy',
        comment: null,
      });

      statusEl.textContent = ok ? '✓ Enviado' : '✗ Erro';
      if (!ok) {
        // allow retry
        primaryBtn.disabled = false;
        secondaryBtn.disabled = false;
      }
    }

    openCommentModal(messageEl, statusEl) {
      const modal = this.modal;
      const overlay = this.modalOverlay;
      if (!modal || !overlay) return;

      // Reset
      this.modalTextarea.value = '';
      this.modalIncludeContext.checked = true;

      overlay.style.display = 'block';
      modal.style.display = 'block';

      this.modalOnSubmit = async () => {
        const comment = this.modalTextarea.value.trim();
        statusEl.style.display = 'inline-flex';
        statusEl.textContent = 'Enviando…';

        const ok = await this.submitFeedback({
          messageEl,
          rating: 3,
          feedback_type: 'general',
          comment: comment || null,
          includeContext: this.modalIncludeContext.checked,
        });

        statusEl.textContent = ok ? '✓ Enviado' : '✗ Erro';
        if (!ok) return;

        this.closeModal();
      };

      this.modalTextarea.focus();
    }

    closeModal() {
      if (this.modalOverlay) this.modalOverlay.style.display = 'none';
      if (this.modal) this.modal.style.display = 'none';
      this.modalOnSubmit = null;
    }

    ensureModalScaffold() {
      // Use admin's modal classes if available, but keep self-contained
      const overlay = document.createElement('div');
      overlay.className = 'modal-overlay';
      overlay.style.display = 'none';
      overlay.addEventListener('click', () => this.closeModal());

      const modal = document.createElement('div');
      modal.className = 'modal chatfb-modal';
      modal.style.display = 'none';
      modal.setAttribute('role', 'dialog');
      modal.setAttribute('aria-modal', 'true');

      modal.innerHTML = `
        <div class="modal-content">
          <div class="modal-header">
            <h3>💬 Feedback</h3>
            <button type="button" class="btn btn-secondary" data-chatfb-close>✕</button>
          </div>
          <div class="modal-body">
            <textarea placeholder="Conte o que aconteceu… (opcional)"></textarea>
            <div class="chatfb-hint">
              <label style="display:flex; gap:8px; align-items:center; margin-top:10px;">
                <input type="checkbox" checked>
                <span>Incluir pergunta e resposta no feedback</span>
              </label>
            </div>
            <div class="chatfb-row">
              <button type="button" class="btn btn-secondary" data-chatfb-cancel>Cancelar</button>
              <button type="button" class="btn btn-primary" data-chatfb-submit>Enviar</button>
            </div>
          </div>
        </div>
      `;

      document.body.appendChild(overlay);
      document.body.appendChild(modal);

      this.modalOverlay = overlay;
      this.modal = modal;
      this.modalTextarea = modal.querySelector('textarea');
      this.modalIncludeContext = modal.querySelector('input[type="checkbox"]');

      modal.querySelector('[data-chatfb-close]').addEventListener('click', () => this.closeModal());
      modal.querySelector('[data-chatfb-cancel]').addEventListener('click', () => this.closeModal());
      modal.querySelector('[data-chatfb-submit]').addEventListener('click', async () => {
        if (typeof this.modalOnSubmit === 'function') {
          await this.modalOnSubmit();
        }
      });

      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') this.closeModal();
      });
    }

    async submitFeedback({ messageEl, rating, feedback_type, comment, includeContext = true }) {
      const traceId = this.getTraceId();
      const messageIndex = Number(messageEl.dataset.chatfbMessageIndex || '0');

      const payload = {
        trace_id: traceId,
        rating: rating,
        feedback_type: feedback_type,
        comment: comment,
        message_index: messageIndex,
      };

      if (includeContext) {
        const q = messageEl.dataset.chatfbQueryText || '';
        const r = messageEl.dataset.chatfbResponseText || '';
        payload.query_text = q || null;
        payload.response_text = r || null;
      }

      try {
        const resp = await fetch(this.apiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        });

        return resp.ok;
      } catch (e) {
        console.error('feedback submit failed', e);
        return false;
      }
    }
  }

  function initWhenReady() {
    // chat-client.js does: window.chatClient = new ChatClient();
    // But if not, we'll still initialize without it.
    const chatClient = window.chatClient || null;
    new FeedbackManager(chatClient);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWhenReady);
  } else {
    initWhenReady();
  }
})();
