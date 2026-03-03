/**
 * Admin API Client
 * ----------------
 * Handles all HTTP communication between the admin SPA and the Resync backend.
 *
 * Authentication strategy:
 *   - The user logs in via POST /auth/login (JSON credentials).
 *   - The server responds with Set-Cookie: access_token=<JWT> (HttpOnly).
 *   - All subsequent requests pass `credentials: 'include'` so the browser
 *     automatically attaches that cookie — no manual Authorization header needed.
 *   - On a 401 response the client clears the session flag and reloads,
 *     showing the login modal again.
 */

class AdminAPIClient {
    constructor() {
        /** @type {string} Base URL of the running Resync server. */
        this.baseUrl = window.location.origin;
    }

    /**
     * Returns extra request headers.
     * Auth is carried by the HttpOnly `access_token` cookie — not by a header.
     *
     * @returns {Record<string, string>} Always an empty object.
     */
    getAuthHeader() {
        // Auth is handled via HttpOnly cookie set at login — no manual header needed
        return {};
    }

    /**
     * Perform an authenticated HTTP request.
     *
     * @param {string} endpoint  - Relative path (e.g. '/admin/config') or full URL.
     * @param {RequestInit} [options] - Optional fetch options (method, body, headers…).
     * @returns {Promise<unknown>}    - Parsed JSON response body.
     * @throws {Error}               - On non-2xx responses or network errors.
     */
    async request(endpoint, options = {}) {
        const url = endpoint.startsWith('http') ? endpoint : `${this.baseUrl}${endpoint}`;

        /** @type {HeadersInit} */
        const headers = {
            'Content-Type': 'application/json',
            ...this.getAuthHeader(),
            ...options.headers,
        };

        try {
            // `credentials: 'include'` ensures the browser sends the access_token cookie.
            const response = await fetch(url, { ...options, headers, credentials: 'include' });

            if (response.status === 401) {
                // Session expired or cookie missing — force fresh login.
                sessionStorage.removeItem('admin_logged_in');
                window.location.reload();
                throw new Error('Unauthorized');
            }

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(error.detail || response.statusText);
            }

            return await response.json();
        } catch (error) {
            console.error(`API Request failed: ${endpoint}`, error);
            throw error;
        }
    }

    /* ── Convenience wrappers ─────────────────────────────────────────────── */

    /** @param {string} endpoint */
    get(endpoint) { return this.request(endpoint, { method: 'GET' }); }

    /**
     * @param {string}  endpoint
     * @param {unknown} data     - Request body (will be JSON-serialised).
     */
    post(endpoint, data) { return this.request(endpoint, { method: 'POST', body: JSON.stringify(data) }); }

    /**
     * @param {string}  endpoint
     * @param {unknown} data     - Request body (will be JSON-serialised).
     */
    put(endpoint, data) { return this.request(endpoint, { method: 'PUT', body: JSON.stringify(data) }); }

    /** @param {string} endpoint */
    delete(endpoint) { return this.request(endpoint, { method: 'DELETE' }); }
}
