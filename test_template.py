import asyncio
from fastapi import FastAPI, Request
from starlette.datastructures import Headers
from resync.app_factory import ApplicationFactory

async def test_render():
    factory = ApplicationFactory()
    app = factory.create_application()
    
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"host", b"localhost")],
    }
    request = Request(scope)
    try:
        response = await factory._render_template("chat.html", request)
        print("Success:", response)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(test_render())
