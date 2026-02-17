/**
 * Admin Feedback Curation Module
 * Handles review and incorporation of user feedback.
 */

class AdminFeedback {
    constructor(app) {
        this.app = app;
        this.api = app.api;
        // Prefix from feedback_curation.py is /api/v1/admin/feedback
        this.basePath = '/api/v1/admin/feedback';
    }

    async loadFeedbackView() {
        const content = document.getElementById('content');

        content.innerHTML = `
            <div class="header-title" style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1>Feedback Curation</h1>
                    <p>Review and incorporate user feedback into the Knowledge Base</p>
                </div>
                <button class="btn btn-neu" onclick="window.feedbackModule.loadStats()">View Stats</button>
            </div>

            <div class="dashboard-grid" id="feedback-stats-summary" style="margin-bottom: 2rem;">
                <!-- Stats loaded dynamically -->
            </div>

            <div class="card">
                <div class="card-title">Pending Feedback</div>
                <div id="feedback-list">
                    <div class="stat-label">Loading feedback...</div>
                </div>
            </div>

            <!-- Review Modal -->
            <div id="reviewModal" class="modal" style="display: none;">
                <div class="modal-content card" style="min-width: 600px;">
                    <h2>Review Feedback</h2>
                    <div id="reviewContent" style="margin-bottom: 1.5rem;"></div>
                    <form id="reviewForm">
                        <input type="hidden" name="id" id="reviewId">
                        
                        <div class="form-group">
                            <label>Correct Answer / Improvement</label>
                            <textarea name="correction" id="reviewCorrection" class="form-input" rows="5" required></textarea>
                            <small style="color: var(--text-secondary);">This text will be learned by the system.</small>
                        </div>
                        
                        <div class="form-group" style="flex-direction: row; gap: 10px; align-items: center;">
                            <input type="checkbox" name="incorporate" id="reviewIncorporate" checked>
                            <label for="reviewIncorporate">Incorporate into Knowledge Base (Golden Record)</label>
                        </div>

                        <div style="margin-top: 1.5rem; display: flex; gap: 10px; justify-content: flex-end;">
                            <button type="button" class="btn btn-neu" onclick="document.getElementById('reviewModal').style.display='none'">Cancel</button>
                            <button type="button" class="btn btn-danger" onclick="window.feedbackModule.rejectCurrent()">Reject</button>
                            <button type="submit" class="btn btn-primary">Approve & Learn</button>
                        </div>
                    </form>
                </div>
            </div>
        `;

        window.feedbackModule = this;
        this.loadStats();
        this.loadPendingList();
    }

    async loadStats() {
        try {
            const stats = await this.api.get(`${this.basePath}/stats`);
            const container = document.getElementById('feedback-stats-summary');
            if (container) {
                container.innerHTML = `
                    <div class="card stat-card">
                        <span class="stat-label">Pending</span>
                        <div class="stat-value">${stats.pending}</div>
                    </div>
                    <div class="card stat-card">
                        <span class="stat-label">With Correction</span>
                        <div class="stat-value">${stats.pending_with_correction}</div>
                    </div>
                     <div class="card stat-card">
                        <span class="stat-label">Approved</span>
                        <div class="stat-value">${stats.approved}</div>
                    </div>
                    <div class="card stat-card">
                        <span class="stat-label">Incorporated</span>
                        <div class="stat-value">${stats.incorporated}</div>
                    </div>
                `;
            }
        } catch (err) {
            console.error('Failed to load stats', err);
        }
    }

    async loadPendingList() {
        const container = document.getElementById('feedback-list');
        try {
            const list = await this.api.get(`${this.basePath}/pending`);

            if (list.length === 0) {
                container.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--text-secondary);">No pending feedback. Good job!</div>';
                return;
            }

            container.innerHTML = `
                <table class="data-table" style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: rgba(0,0,0,0.02); border-bottom: 1px solid var(--border);">
                            <th style="padding: 1rem; text-align: left;">ID</th>
                            <th style="padding: 1rem; text-align: left;">Query / Response</th>
                            <th style="padding: 1rem; text-align: left;">Rating</th>
                            <th style="padding: 1rem; text-align: left;">User Feedback</th>
                            <th style="padding: 1rem; text-align: right;">Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${list.map(item => this.renderRow(item)).join('')}
                    </tbody>
                </table>
            `;
        } catch (err) {
            container.innerHTML = `<div class="stat-label">Error: ${err.message}</div>`;
        }
    }

    renderRow(item) {
        return `
            <tr>
                <td style="padding: 1rem; border-bottom: 1px solid var(--border); vertical-align: top;">${item.id}</td>
                <td style="padding: 1rem; border-bottom: 1px solid var(--border); vertical-align: top;">
                    <div style="font-weight: 500; margin-bottom: 5px;">Q: ${item.query_text || '-'}</div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">A: ${item.response_text || '-'}</div>
                </td>
                <td style="padding: 1rem; border-bottom: 1px solid var(--border); vertical-align: top;">
                    <span class="badge ${item.rating > 2 ? 'badge-success' : 'badge-error'}">${item.rating}/5</span>
                </td>
                <td style="padding: 1rem; border-bottom: 1px solid var(--border); vertical-align: top;">
                    ${item.feedback_text ? `<div>${item.feedback_text}</div>` : '<em style="color:#999">No text</em>'}
                </td>
                <td style="padding: 1rem; text-align: right; border-bottom: 1px solid var(--border); vertical-align: top;">
                    <button class="btn btn-primary" onclick="window.feedbackModule.openReviewModal(${item.id})">Review</button>
                </td>
            </tr>
        `;
    }

    async openReviewModal(id) {
        try {
            const detail = await this.api.get(`${this.basePath}/${id}`);
            const modal = document.getElementById('reviewModal');
            document.getElementById('reviewContent').innerHTML = `
                <div style="background: rgba(0,0,0,0.03); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <strong>Query:</strong> ${detail.query_text}<br><br>
                    <strong>Current Response:</strong><br>
                    <div style="font-family: monospace; font-size: 0.9rem; margin-top: 5px;">${detail.response_text}</div>
                </div>
                <div>
                    <strong>User Feedback:</strong> ${detail.feedback_text || 'None'} (${detail.rating}/5)
                </div>
            `;

            document.getElementById('reviewId').value = id;
            document.getElementById('reviewCorrection').value = detail.feedback_text || detail.response_text || ''; // Pre-fill

            modal.style.display = 'flex';

            document.getElementById('reviewForm').onsubmit = (e) => this.handleApprove(e);

        } catch (err) {
            alert(`Error loading details: ${err.message}`);
        }
    }

    async handleApprove(e) {
        e.preventDefault();
        const id = document.getElementById('reviewId').value;
        const correction = document.getElementById('reviewCorrection').value;
        const incorporate = document.getElementById('reviewIncorporate').checked;

        try {
            await this.api.post(`${this.basePath}/${id}/approve`, {
                reviewer_id: 'admin', // In real app, get from auth
                user_correction: correction,
                incorporate_to_kb: incorporate
            });

            document.getElementById('reviewModal').style.display = 'none';
            alert('Feedback approved and learned!');
            this.loadPendingList();
            this.loadStats();
        } catch (err) {
            alert(`Error approving: ${err.message}`);
        }
    }

    async rejectCurrent() {
        const id = document.getElementById('reviewId').value;
        const reason = prompt("Reason for rejection:");
        if (!reason) return;

        try {
            await this.api.post(`${this.basePath}/${id}/reject`, {
                reviewer_id: 'admin',
                reason: reason
            });

            document.getElementById('reviewModal').style.display = 'none';
            this.loadPendingList();
            this.loadStats();
        } catch (err) {
            alert(`Error rejecting: ${err.message}`);
        }
    }
}
