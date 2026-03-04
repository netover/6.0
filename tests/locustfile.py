"""
Locust Load Test for Resync API
================================
Usage:
    1. Start the server: python -m uvicorn resync.main:app --host 0.0.0.0 --port 8000
    2. Run Locust: locust -f tests/locustfile.py --host=http://localhost:8000
    3. Open http://localhost:8089 for the web interface
"""

import random
from locust import HttpUser, task, between, events


class ResyncUser(HttpUser):
    """Simulates a typical user interacting with the Resync API"""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts - perform login if needed"""
        # Try to access the root to get initial cookies/session
        self.client.get("/")
    
    @task(3)
    def health_check(self):
        """Check API health - frequent operation"""
        self.client.get("/api/health")
    
    @task(2)
    def chat_message(self):
        """Send a chat message via WebSocket (HTTP fallback for testing)"""
        # Note: WebSocket testing requires special handling in Locust
        # This is an HTTP fallback to test the chat endpoint
        headers = {"Content-Type": "application/json"}
        payload = {
            "message": "Hello, test message " + str(random.randint(1, 1000)),
            "agent_id": "tws-general"
        }
        with self.client.post(
            "/api/chat/message",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            elif response.status_code == 401:
                response.failure("Unauthorized - need login")
            else:
                response.failure(f"Got status {response.status_code}")
    
    @task(1)
    def admin_dashboard(self):
        """Access admin dashboard"""
        self.client.get("/admin/", name="/admin/")
    
    @task(1)
    def api_routes(self):
        """Test various API routes"""
        routes = [
            "/api/v1/metrics",
            "/api/v1/status",
        ]
        for route in routes:
            self.client.get(route, name="/api/v1/[route]")


class AdminUser(HttpUser):
    """Simulates an admin user with more intensive operations"""
    
    wait_time = between(2, 5)
    
    @task(2)
    def admin_settings(self):
        """Access admin settings"""
        self.client.get("/api/admin/settings", name="/api/admin/settings")
    
    @task(1)
    def admin_users(self):
        """List users"""
        self.client.get("/api/admin/users", name="/api/admin/users")
    
    @task(1)
    def admin_stats(self):
        """Get system stats"""
        self.client.get("/api/admin/stats", name="/api/admin/stats")


class HeavyUser(HttpUser):
    """Simulates a heavy user doing RAG operations"""
    
    wait_time = between(5, 10)
    
    @task(1)
    def rag_query(self):
        """Test RAG query endpoint"""
        payload = {
            "query": "test query " + str(random.randint(1, 100)),
            "top_k": 5
        }
        with self.client.post(
            "/api/rag/query",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"RAG query failed: {response.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts"""
    print("🚀 Load test starting...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops"""
    print("🛑 Load test finished!")
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Total failures: {environment.stats.total.num_failures}")
