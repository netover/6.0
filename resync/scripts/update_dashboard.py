# ruff: noqa: E501
"""Utility script to patch websocket auth flow in monitoring_dashboard.py."""

from pathlib import Path

TARGET = Path("resync/api/monitoring_dashboard.py")

OLD_WS = '''@router.websocket("/ws")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket para métricas em tempo real com autenticação."""
    username = await _verify_ws_admin(websocket)
    if not username:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    if not await ws_manager.connect(websocket):
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER); return'''

NEW_WS = '''@router.websocket("/ws")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket para métricas em tempo real com autenticação."""
    await websocket.accept()
    username = await _verify_ws_admin(websocket)
    if not username:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    if not await ws_manager.connect(websocket):
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER); return'''


def main() -> None:
    content = TARGET.read_text(encoding="utf-8")
    updated = content.replace(OLD_WS, NEW_WS)
    TARGET.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
