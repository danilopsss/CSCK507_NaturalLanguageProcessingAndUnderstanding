import torch
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pathlib import Path


app = FastAPI()

ATTENTION_MODEL = next(Path(__file__).parent.parent.rglob("**/*/*with_attention*"))
NO_ATTENTION_MODEL = next(Path(__file__).parent.parent.rglob("**/*/*no_attention*"))

templates = next(Path(__file__).parent.rglob("**/*/templates"))
templates_path = templates.as_posix()

static = next(Path(__file__).parent.rglob("**/*/static"))
static_path = static.as_posix()

templates = Jinja2Templates(directory=templates_path)

app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )



@app.websocket("/attention")
async def websocket_attention_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(
            f"Machine: {data}"
        )

@app.websocket("/noAttention")
async def websocket_no_attention_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(
            f"Machine: {data}"
        )
