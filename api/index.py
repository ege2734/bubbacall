import json
from typing import List

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from google import genai
from google.genai.types import UsageMetadata
from pydantic import BaseModel

from .utils.prompt import ClientMessage
from .utils.settings import get_setting

app = FastAPI()

model = "gemini-2.0-flash-live-001"
config = {"response_modalities": ["TEXT"]}

client = genai.Client(api_key=get_setting("GEMINI_API_KEY"))


class Request(BaseModel):
    messages: List[ClientMessage]


def create_text_response(text: str):
    return f"0:{json.dumps(text)}\n".encode("utf-8")


def create_end_response(finish_reason: str, usage_metadata: UsageMetadata):
    return_dict = {
        "finishReason": finish_reason,
        "usage": {
            "promptTokens": usage_metadata.prompt_token_count,
            "completionTokens": usage_metadata.response_token_count,
        },
        "isContinued": False,
    }
    return f"d:{json.dumps(return_dict)}\n".encode("utf-8")


async def do_stream(messages: List[ClientMessage]):
    async with client.aio.live.connect(model=model, config=config) as session:
        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": messages[-1].content}]},
            turn_complete=True,
        )
        async for response in session.receive():
            print(response)
            if response.text is not None:
                yield create_text_response(response.text)

            if response.server_content.turn_complete:
                yield create_end_response(
                    finish_reason="stop",
                    usage_metadata=response.usage_metadata,
                )


@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query("data")):
    response = StreamingResponse(do_stream(request.messages))

    response.headers["x-vercel-ai-data-stream"] = "v1"
    return response
