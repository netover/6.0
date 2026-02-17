/**
 * Security Utilities for XSS Prevention
 * Provides functions for sanitizing user-provided data before rendering in HTML
 */

/**
 * Escapes HTML special characters to prevent XSS attacks
 * @param {string} str - The string to escape
 * @returns {string} The escaped string safe for innerHTML
 */
function escapeHtml(str) {
    if (str === null || str === undefined) {
        return '';
    }
    const strType = typeof str;
    if (strType === 'number' || strType === 'boolean') {
        return String(str);
    }
    if (strType !== 'string') {
        str = String(str);
    }
    return str
        .replace(/&/g, '&')
        .replace(/</g, '<')
        .replace(/>/g, '>')
        .replace(/"/g, '"')
        .replace(/'/g, '&#039;');
}

/**
 * Escapes HTML and preserves newlines by converting them to <br> tags
 * @param {string} str - The string to escape
 * @returns {string} The escaped string with line breaks
 */
function escapeHtmlWithNewlines(str) {
    if (str === null || str === undefined) {
        return '';
    }
    return escapeHtml(str).replace(/\n/g, '<br>');
}

/**
 * Sanitizes an object by escaping all string values
 * @param {Object} obj - The object to sanitize
 * @returns {Object} A new object with escaped string values
 */
function sanitizeObject(obj) {
    if (obj === null || obj === undefined) {
        return obj;
    }
    if (typeof obj !== 'object') {
        return escapeHtml(obj);
    }
    const result = Array.isArray(obj) ? [] : {};
    for (const key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
            const value = obj[key];
            if (typeof value === 'string') {
                result[key] = escapeHtml(value);
            } else if (typeof value === 'object' && value !== null) {
                result[key] = sanitizeObject(value);
            } else {
                result[key] = value;
            }
        }
    }
    return result;
}

// Export for module systems if available
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { escapeHtml, escapeHtmlWithNewlines, sanitizeObject };
}
