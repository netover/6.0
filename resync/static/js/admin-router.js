/**
 * Admin Router - Handles SPA navigation using hash routing.
 */
class AdminRouter {
    constructor(app) {
        this.app = app;
        this.routes = {};
    }

    addRoute(hash, handler) {
        this.routes[hash] = handler;
    }

    navigate(hash) {
        console.log(`Navigating to: ${hash}`);
        const handler = this.routes[hash] || this.routes['health'];

        // Update UI
        this.updateActiveLink(hash);

        if (handler) {
            handler();
        }
    }

    updateActiveLink(hash) {
        document.querySelectorAll('.nav-link').forEach(link => {
            if (link.getAttribute('href') === `#${hash}`) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }
}
