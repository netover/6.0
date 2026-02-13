// =============================================================================
// Resync TWS Monitor - Service Worker
// Gerencia notificações push e cache offline
// =============================================================================

const CACHE_NAME = 'resync-tws-v1';
const STATIC_ASSETS = [
    '/',
    '/static/css/dashboard.css',
    '/static/js/dashboard.js',
    '/templates/realtime_dashboard.html',
];

// =============================================================================
// INSTALLATION
// =============================================================================

self.addEventListener('install', (event) => {
    console.log('[SW] Installing service worker...');
    
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('[SW] Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            })
            .then(() => {
                console.log('[SW] Installation complete');
                return self.skipWaiting();
            })
    );
});

// =============================================================================
// ACTIVATION
// =============================================================================

self.addEventListener('activate', (event) => {
    console.log('[SW] Activating service worker...');
    
    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames
                        .filter((name) => name !== CACHE_NAME)
                        .map((name) => {
                            console.log('[SW] Removing old cache:', name);
                            return caches.delete(name);
                        })
                );
            })
            .then(() => {
                console.log('[SW] Activation complete');
                return self.clients.claim();
            })
    );
});

// =============================================================================
// FETCH HANDLER
// =============================================================================

self.addEventListener('fetch', (event) => {
    // Skip non-GET requests
    if (event.request.method !== 'GET') {
        return;
    }
    
    // Skip WebSocket connections
    if (event.request.url.includes('/ws')) {
        return;
    }
    
    // Skip API calls (we want fresh data)
    if (event.request.url.includes('/api/')) {
        return;
    }
    
    event.respondWith(
        caches.match(event.request)
            .then((cachedResponse) => {
                if (cachedResponse) {
                    return cachedResponse;
                }
                
                return fetch(event.request)
                    .then((response) => {
                        // Don't cache if not a valid response
                        if (!response || response.status !== 200) {
                            return response;
                        }
                        
                        // Clone the response
                        const responseToCache = response.clone();
                        
                        caches.open(CACHE_NAME)
                            .then((cache) => {
                                cache.put(event.request, responseToCache);
                            });
                        
                        return response;
                    });
            })
    );
});

// =============================================================================
// PUSH NOTIFICATIONS
// =============================================================================

self.addEventListener('push', (event) => {
    console.log('[SW] Push notification received');
    
    let data = {
        title: 'Resync TWS Alert',
        body: 'Novo evento no TWS',
        icon: '/static/icons/icon-192.png',
        badge: '/static/icons/badge-72.png',
        tag: 'resync-alert',
        data: {},
    };
    
    if (event.data) {
        try {
            const payload = event.data.json();
            data = {
                ...data,
                ...payload,
            };
        } catch (e) {
            data.body = event.data.text();
        }
    }
    
    const options = {
        body: data.body,
        icon: data.icon,
        badge: data.badge,
        tag: data.tag,
        data: data.data,
        vibrate: [200, 100, 200],
        requireInteraction: data.severity === 'critical',
        actions: getNotificationActions(data.severity),
    };
    
    event.waitUntil(
        self.registration.showNotification(data.title, options)
    );
});

function getNotificationActions(severity) {
    const actions = [
        {
            action: 'view',
            title: 'Ver Detalhes',
            icon: '/static/icons/view.png',
        },
    ];
    
    if (severity === 'critical' || severity === 'error') {
        actions.push({
            action: 'acknowledge',
            title: 'Reconhecer',
            icon: '/static/icons/check.png',
        });
    }
    
    return actions;
}

// =============================================================================
// NOTIFICATION CLICK
// =============================================================================

self.addEventListener('notificationclick', (event) => {
    console.log('[SW] Notification clicked:', event.action);
    
    event.notification.close();
    
    const data = event.notification.data || {};
    
    if (event.action === 'view' || !event.action) {
        // Open or focus the dashboard
        event.waitUntil(
            clients.matchAll({ type: 'window', includeUncontrolled: true })
                .then((clientList) => {
                    // Try to focus existing window
                    for (const client of clientList) {
                        if (client.url.includes('/dashboard') && 'focus' in client) {
                            return client.focus();
                        }
                    }
                    
                    // Open new window if none exists
                    if (clients.openWindow) {
                        let url = '/templates/realtime_dashboard.html';
                        
                        // Add event ID to URL if available
                        if (data.event_id) {
                            url += `?event=${data.event_id}`;
                        }
                        
                        return clients.openWindow(url);
                    }
                })
        );
    }
    
    if (event.action === 'acknowledge') {
        // Send acknowledgment to server
        event.waitUntil(
            fetch('/api/v1/monitoring/events/acknowledge', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    event_id: data.event_id,
                    acknowledged_at: new Date().toISOString(),
                }),
            })
        );
    }
});

// =============================================================================
// BACKGROUND SYNC
// =============================================================================

self.addEventListener('sync', (event) => {
    console.log('[SW] Background sync:', event.tag);
    
    if (event.tag === 'sync-settings') {
        event.waitUntil(syncSettings());
    }
    
    if (event.tag === 'sync-acknowledgments') {
        event.waitUntil(syncAcknowledgments());
    }
});

async function syncSettings() {
    try {
        const settings = await getStoredSettings();
        if (settings) {
            await fetch('/api/v1/monitoring/config', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(settings),
            });
            console.log('[SW] Settings synced');
        }
    } catch (e) {
        console.error('[SW] Settings sync failed:', e);
    }
}

async function syncAcknowledgments() {
    try {
        const acks = await getStoredAcknowledgments();
        if (acks && acks.length > 0) {
            await fetch('/api/v1/monitoring/events/acknowledge/batch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ acknowledgments: acks }),
            });
            await clearStoredAcknowledgments();
            console.log('[SW] Acknowledgments synced');
        }
    } catch (e) {
        console.error('[SW] Acknowledgments sync failed:', e);
    }
}

// =============================================================================
// INDEXED DB HELPERS
// =============================================================================

const DB_NAME = 'resync-sw-db';
const DB_VERSION = 1;

function openDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
        
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            
            if (!db.objectStoreNames.contains('settings')) {
                db.createObjectStore('settings', { keyPath: 'id' });
            }
            
            if (!db.objectStoreNames.contains('acknowledgments')) {
                db.createObjectStore('acknowledgments', { keyPath: 'event_id' });
            }
        };
    });
}

async function getStoredSettings() {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction('settings', 'readonly');
        const store = tx.objectStore('settings');
        const request = store.get('current');
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result?.data);
    });
}

async function getStoredAcknowledgments() {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction('acknowledgments', 'readonly');
        const store = tx.objectStore('acknowledgments');
        const request = store.getAll();
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
    });
}

async function clearStoredAcknowledgments() {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction('acknowledgments', 'readwrite');
        const store = tx.objectStore('acknowledgments');
        const request = store.clear();
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve();
    });
}

// =============================================================================
// MESSAGE HANDLER
// =============================================================================

self.addEventListener('message', (event) => {
    console.log('[SW] Message received:', event.data);
    
    if (event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
    
    if (event.data.type === 'GET_VERSION') {
        event.ports[0].postMessage({
            version: CACHE_NAME,
        });
    }
    
    if (event.data.type === 'CLEAR_CACHE') {
        caches.delete(CACHE_NAME).then(() => {
            event.ports[0].postMessage({ success: true });
        });
    }
});

console.log('[SW] Service worker loaded');
