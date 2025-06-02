import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Query, WebSocket
from fastapi.responses import StreamingResponse
from google import genai
from mcp import ClientSessionGroup
from pydantic import BaseModel
from twilio.rest import Client as TwilioClient

from api.utils.chat import do_stream
from api.utils.mcp_util import google_maps
from api.utils.mongodb import MongoDB
from api.utils.prompt import ClientMessage
from api.utils.settings import get_setting
from api.utils.twilio_phone_call import stream_call as stream_twilio_call

mcp_session_group: ClientSessionGroup | None = None
mongodb_client: MongoDB | None = None
twilio_client: TwilioClient | None = None
gemini_client: genai.Client | None = None

FAKE_PHONE_CALL = True


# Based on https://fastapi.tiangolo.com/advanced/events/#lifespan.
@asynccontextmanager
async def lifespan(_: FastAPI):
    global mcp_session_group, mongodb_client, twilio_client, gemini_client
    assert mcp_session_group is None, "Sessions already initialized?"
    params = [google_maps()]
    async with ClientSessionGroup() as group:
        for param in params:
            await group.connect_to_server(param)
        mcp_session_group = group
        mongodb_client = MongoDB()
        twilio_client = TwilioClient(
            get_setting("TWILIO_ACCOUNT_SID"), get_setting("TWILIO_AUTH_TOKEN")
        )
        gemini_client = genai.Client(api_key=get_setting("GEMINI_API_KEY"))

        yield
        mcp_session_group = None
        mongodb_client = None
        twilio_client = None


app = FastAPI(lifespan=lifespan)

# Add basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Request(BaseModel):
    messages: List[ClientMessage]


@app.websocket("/task-stream/{task_id}")
async def task_stream(websocket: WebSocket, task_id: str):
    logging.info(f"Task stream connected: {task_id}")
    await websocket.accept()
    task = await mongodb_client.get_task(task_id)

    assert task is not None

    await stream_twilio_call(mongodb_client, websocket, task, task_id)


@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query("data")):
    assert protocol is not None
    response = StreamingResponse(
        do_stream(
            gemini_client=gemini_client,
            mcp_session_group=mcp_session_group,
            twilio_client=twilio_client,
            mongodb_client=mongodb_client,
            messages=request.messages,
        )
    )

    response.headers["x-vercel-ai-data-stream"] = "v1"
    return response
