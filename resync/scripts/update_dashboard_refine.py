# ruff: noqa: E501
with open("resync/api/monitoring_dashboard.py", "r") as f:
    content = f.read()

# 3.5 Move late imports
content = "from resync.core.metrics import runtime_metrics\n" + content
content = content.replace("    from resync.core.metrics import runtime_metrics", "")

# 3.2 Update _empty_history
content = content.replace(
    'return {"timestamps": [], "api": {"requests_per_sec": [], "error_rate": []}, "sample_count": 0}',
    'return {"timestamps": [], "api": {"requests_per_sec": [], "error_rate": []}, "cache": {"hit_ratio": []}, "agents": {"active": []}, "sample_count": 0}',
)

# 3.3 Improve get_current_metrics
content = content.replace(
    'if not raw: return self._empty_response("initializing")',
    'if raw is None: return self._empty_response("initializing")',
)

# 3.3 Improve add_error_sample
old_add_error = """    async def add_error_sample(self, error: Exception) -> None:
        \"\"\"Persiste amostra indicando falha na coleta.\"\"\"
        now_wall = time.time()
        dt_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        error_msg = str(error)[:197] + "..." if len(str(error)) > 200 else str(error)
        global_uptime = await self.get_global_uptime()

        sample = MetricSample(
            timestamp=now_wall, datetime_str=dt_str, collection_error=error_msg,
            system_uptime=global_uptime, system_availability=0.0,
            tws_connected=True  # Evita alerta falso
        )
        await self.add_sample(sample)"""

new_add_error = """    async def add_error_sample(self, error: Exception) -> None:
        \"\"\"Persiste amostra indicando falha na coleta.\"\"\"
        now_wall = time.time()
        dt_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        raw_msg = str(error)
        error_msg = raw_msg[:197] + "..." if len(raw_msg) > 200 else raw_msg
        global_uptime = await self.get_global_uptime()

        # Tenta determinar conexão real se possível, senão assume falso
        tws_status = False
        try:
            snapshot = runtime_metrics.get_snapshot()
            tws_status = snapshot.get("slo", {}).get("tws_connection_success_rate", 0) > 0.5
        except Exception: pass

        sample = MetricSample(
            timestamp=now_wall, datetime_str=dt_str, collection_error=error_msg,
            system_uptime=global_uptime, system_availability=0.0,
            tws_connected=tws_status
        )
        await self.add_sample(sample)"""
content = content.replace(old_add_error, new_add_error)

# 3.3 Improve get_global_uptime
old_uptime = """    async def get_global_uptime(self) -> float:
        redis = get_redis_client()
        try:
            now = time.time()
            was_set = await redis.set(REDIS_KEY_START_TIME, str(now), nx=True)
            raw = await redis.get(REDIS_KEY_START_TIME)
            return now - float(raw or now)
        except Exception: return 0.0"""

new_uptime = """    async def get_global_uptime(self) -> float:
        redis = get_redis_client()
        try:
            now = time.time()
            await redis.set(REDIS_KEY_START_TIME, str(now), nx=True)
            raw = await redis.get(REDIS_KEY_START_TIME)
            return now - float(raw or now)
        except Exception:
            logger.debug("Falha ao obter uptime global do Redis")
            return 0.0"""
content = content.replace(old_uptime, new_uptime)

# 3.4 Fix start_sync idempotency
old_start_sync = """    async def start_sync(self):
        if self._stop_event.is_set():
            return
        if self._sync_task is None or self._sync_task.done():
            self._sync_task = asyncio.create_task(self._pubsub_listener())"""

new_start_sync = """    async def start_sync(self):
        if self._sync_task is None or self._sync_task.done():
            self._stop_event.clear()
            self._sync_task = asyncio.create_task(self._pubsub_listener())"""
content = content.replace(old_start_sync, new_start_sync)

# 3.5 Improve WebSocketManager.connect logging
old_connect = """    async def connect(self, websocket: WebSocket) -> bool:
        async with self._lock:
            if len(self._clients) >= MAX_WS_CONNECTIONS: return False
            self._clients.add(websocket)
            return True"""

new_connect = """    async def connect(self, websocket: WebSocket) -> bool:
        async with self._lock:
            if len(self._clients) >= MAX_WS_CONNECTIONS:
                logger.warning("Limite de conexões WebSocket atingido: %d", MAX_WS_CONNECTIONS)
                return False
            self._clients.add(websocket)
            return True"""
content = content.replace(old_connect, new_connect)

# 3.5 Improve _safe_send logging
content = content.replace(
    'logger.exception("Falha ao enviar mensagem WebSocket; desconectando cliente")',
    'logger.debug("Falha ao enviar mensagem WebSocket; desconectando cliente")',
)

# 3.1 Fix lock management in collect_metrics_sample
old_collect = """async def collect_metrics_sample() -> None:
    \"\"\"Apenas um worker coleta por vez (Liderança via Redis Lock).\"\"\"
    redis = get_redis_client()
    if not await redis.set(REDIS_LOCK_COLLECTOR, "leader", ex=8, nx=True):
        return


    try:
        snapshot = runtime_metrics.get_snapshot()"""

new_collect = """async def collect_metrics_sample() -> None:
    \"\"\"Apenas um worker coleta por vez (Liderança via Redis Lock).\"\"\"
    redis = get_redis_client()
    # Aumentado TTL para 15s para maior segurança contra sobreposição
    if not await redis.set(REDIS_LOCK_COLLECTOR, "leader", ex=15, nx=True):
        return

    try:
        snapshot = runtime_metrics.get_snapshot()"""

content = content.replace(old_collect, new_collect)

# Add finally block to release lock
# This is tricky because the try block ends far down.
# I'll search for the end of the try block in collect_metrics_sample.

# Let's use a regex to find the try block in collect_metrics_sample and add the finally block.
# Original:
#    try:
#        ...
#    except Exception as e:
#        ...
#        await redis.publish(REDIS_CH_BROADCAST, json_dumps(current))

# To be safe, I'll replace the whole function.

old_collect_func = r"async def collect_metrics_sample\(\) -> None:.*?await redis\.publish\(REDIS_CH_BROADCAST, json_dumps\(current\)\)"
# Wait, there's another publish in the try block.

# I'll re-read the file to be sure about the structure of collect_metrics_sample
with open("resync/api/monitoring_dashboard.py", "w") as f:
    f.write(content)
