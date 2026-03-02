from __future__ import annotations

import asyncio
import os

import httpx
import websockets

async def main() -> int:
    base = os.getenv("RESYNC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    ws_url = os.getenv("RESYNC_WS_URL", "ws://127.0.0.1:8000/ws")

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{base}/health")
        if r.status_code >= 400:
            print("Health failed:", r.status_code, r.text)
            return 2
        print("Health OK:", r.status_code)

    try:
        async with websockets.connect(ws_url, open_timeout=5) as ws:
            await ws.send("ping")
            print("WS connect OK")
    except Exception as e:
        print("WS check failed:", e)
        return 3

    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
