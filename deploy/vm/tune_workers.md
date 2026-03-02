# Worker tuning (Gunicorn + uvicorn-worker)

This deployment uses **Gunicorn** with the **uvicorn-worker** worker class as recommended in the Uvicorn deployment docs. citeturn0search0turn0search2turn0search3

## Key env vars (systemd EnvironmentFile)
- `WEB_CONCURRENCY`: number of workers (default 4)
- `GUNICORN_TIMEOUT`: hard request timeout (default 60s)
- `GUNICORN_GRACEFUL_TIMEOUT`: graceful shutdown window (default 30s)
- `GUNICORN_KEEPALIVE`: keep-alive seconds (default 5)

## Rules of thumb
- Start with: `WEB_CONCURRENCY = (2 * CPU_CORES) + 1` for mixed I/O workloads, then load-test.
- If you have heavy CPU tasks in-request, move them out (background tasks/queues) and keep workers moderate.
