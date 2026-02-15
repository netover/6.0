import re

with open('resync/api/monitoring_dashboard.py', 'r') as f:
    content = f.read()

# Update websocket_metrics
old_ws = """@router.websocket("/ws")
async def websocket_metrics(websocket: WebSocket):
    \"\"\"WebSocket para métricas em tempo real com autenticação.\"\"\"
    username = await _verify_ws_admin(websocket)
    if not username:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    if not await ws_manager.connect(websocket):
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER); return"""

new_ws = """@router.websocket("/ws")
async def websocket_metrics(websocket: WebSocket):
    \"\"\"WebSocket para métricas em tempo real com autenticação.\"\"\"
    await websocket.accept()
    username = await _verify_ws_admin(websocket)
    if not username:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    if not await ws_manager.connect(websocket):
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER); return"""

content = content.replace(old_ws, new_ws)

with open('resync/api/monitoring_dashboard.py', 'w') as f:
    f.write(content)
