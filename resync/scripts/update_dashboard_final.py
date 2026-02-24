# ruff: noqa: E501
import re

with open("resync/api/monitoring_dashboard.py", "r") as f:
    content = f.read()

# 1. Update add_error_sample
add_error_pattern = r"async def add_error_sample\(self, error: Exception\) -> None:.*?await self\.add_sample\(sample\)"
new_add_error = """async def add_error_sample(self, error: Exception) -> None:
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

content = re.sub(add_error_pattern, new_add_error, content, flags=re.DOTALL)

# 2. Update collect_metrics_sample
collect_pattern = r"async def collect_metrics_sample\(\) -> None:.*?await redis\.publish\(REDIS_CH_BROADCAST, json_dumps\(current\)\)"
new_collect = """async def collect_metrics_sample() -> None:
    \"\"\"Apenas um worker coleta por vez (Liderança via Redis Lock).\"\"\"
    redis = get_redis_client()
    # Aumentado TTL para 15s para maior segurança contra sobreposição
    if not await redis.set(REDIS_LOCK_COLLECTOR, "leader", ex=15, nx=True):
        return

    try:
        snapshot = runtime_metrics.get_snapshot()
        now_mono = time.monotonic()
        now_wall = time.time()
        dt_str = datetime.now(timezone.utc).strftime("%H:%M:%S")

        agent = snapshot.get("agent", {})
        slo = snapshot.get("slo", {})
        req_total = _safe_int(agent.get("initializations"))
        uptime = await _metrics_store.get_global_uptime()

        def build_sample(rps: float) -> MetricSample:
            return MetricSample(
                timestamp=now_wall, datetime_str=dt_str,
                requests_total=req_total, requests_per_sec=rps,
                error_rate=_safe_float(slo.get("api_error_rate")) * 100,
                tws_connected=_safe_float(slo.get("tws_connection_success_rate")) > 0.5,
                system_uptime=uptime, system_availability=_safe_float(slo.get("availability"), 1.0) * 100
            )

        await _metrics_store.compute_rate_and_add_sample(req_total, now_mono, build_sample)

        current = await _metrics_store.get_current_metrics()
        subscribers = await redis.publish(REDIS_CH_BROADCAST, json_dumps(current))
        if subscribers == 0: logger.debug("Nenhum subscriber no canal de broadcast")

    except Exception as e:
        logger.error("Erro na coleta: %s", e)
        try:
            await _metrics_store.add_error_sample(e)
            current = await _metrics_store.get_current_metrics()
            await redis.publish(REDIS_CH_BROADCAST, json_dumps(current))
        except Exception:
            logger.debug("Falha ao persistir/broadcast amostra de erro")
    finally:
        # Libera o lock explicitamente após a coleta
        await redis.delete(REDIS_LOCK_COLLECTOR)"""

content = re.sub(collect_pattern, new_collect, content, flags=re.DOTALL)

# 3. Update get_global_uptime for better exception handling and unused variable
uptime_pattern = r"async def get_global_uptime\(self\) -> float:.*?return 0\.0"
new_uptime = """async def get_global_uptime(self) -> float:
        redis = get_redis_client()
        try:
            now = time.time()
            await redis.set(REDIS_KEY_START_TIME, str(now), nx=True)
            raw = await redis.get(REDIS_KEY_START_TIME)
            return now - float(raw or now)
        except Exception:
            logger.debug("Falha ao obter uptime global do Redis")
            return 0.0"""
content = re.sub(uptime_pattern, new_uptime, content, flags=re.DOTALL)

with open("resync/api/monitoring_dashboard.py", "w") as f:
    f.write(content)
