import spacy

from fastapi import Depends, FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.utils.tokenizers import DefaultTokenizers
from src.utils.chat import Chat
from src.utils.device import device
from src.dependencies.attention_dependencies import (
    attention_encoder, attention_decoder
)
from src.dependencies.no_attention_dependencies import (
    no_attention_encoder, no_attention_decoder
)


from pathlib import Path

spacy_en = spacy.load("en_core_web_lg")
tokenizer = DefaultTokenizers(spacy_en).default_tokenizer

app = FastAPI()

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
async def websocket_attention_endpoint(
    websocket: WebSocket,
    encode=Depends(attention_encoder),
    decoder=Depends(attention_decoder)
):
    chat = Chat(
        encoder=encode,
        decoder=decoder,
        device=device,
        tokenizer=tokenizer
    )
    await websocket.accept()
    while True:
        text = await websocket.receive_text()
        response = chat.ask(text.strip())
        await websocket.send_text(response)


@app.websocket("/noAttention")
async def websocket_no_attention_endpoint(
    websocket: WebSocket,
    encoder=Depends(no_attention_encoder),
    decoder=Depends(no_attention_decoder)
):
    chat = Chat(
        encoder=encoder,
        decoder=decoder,
        device=device,
        tokenizer=tokenizer
    )
    await websocket.accept()
    while True:
        text = await websocket.receive_text()
        response = chat.ask(text.strip())
        await websocket.send_text(f"Machine: {response}")
