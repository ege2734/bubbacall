import json
import logging
from typing import List

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from google import genai
from google.genai.types import (
    Content,
    GenerateContentConfig,
    LiveConnectConfig,
    Modality,
    Part,
    UsageMetadata,
)
from pydantic import BaseModel

from .utils.phone_call import Task, make_phone_call
from .utils.prompt import ClientMessage, convert_to_gemini_messages
from .utils.settings import get_setting

app = FastAPI()

# Add basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BEGIN_TASK_DEFINITION = "--- BEGIN TASK DEFINITION ---"
END_TASK_DEFINITION = "--- END TASK DEFINITION ---"
SYSTEM_INSTRUCTION = """
You are a helpful AI assistant. You are talking to a user who should give you the following info:
- The name of a business
- The phone number of the business
- A task that the user wants completed by calling the business.

If the user does not provide the required info, you should ask them for it.
If the user asks for a task that is not possible to complete via a phone call, you should
reject and inform the user that you can only take on tasks that are possible to complete via a phone call.

If the task is very vague, you are welcome to ask clarifying questions to the user.

Once you have the required info, confirm to the user the info you have and ask them if they would like to proceed with the task.
If they reject, you can go back to the beginning of the conversation.

If they accept, let the user know you're taking on the task and will let them know when you're done.
"""

TASK_SYSTEM_INSTRUCTION = """
You are a helpful AI assistant. You are talking to a user who should give you the following info:
- The name of a business
- The phone number of the business
- A task that the user wants completed by calling the business.

If the user does not provide the required info, you should ask them for it.
If the user asks for a task that is not possible to complete via a phone call, you should
reject and inform the user that you can only take on tasks that are possible to complete via a phone call.

If the task is very vague, you are welcome to ask clarifying questions to the user.

Once you have the required info, confirm to the user the info you have and ask them if they would like to proceed with the task.
If they reject, you can go back to the beginning of the conversation.

If they accept, your next message should be in the json format provided alongside your config.

Your response should have .task set to 'None' UNTIL the user confirms the task
"""

live_model = "gemini-2.0-flash-live-001"
task_model = "gemini-2.0-flash"
# config = LiveConnectConfig(
#     system_instruction=Content(role="system", parts=[Part(text=SYSTEM_INSTRUCTION)]),
#     response_modalities=[Modality.TEXT],
# )

client = genai.Client(api_key=get_setting("GEMINI_API_KEY"))


class TaskOrNone(BaseModel):
    task: Task | None = None


class Request(BaseModel):
    messages: List[ClientMessage]


def create_config():
    return LiveConnectConfig(
        system_instruction=Content(
            role="system", parts=[Part(text=SYSTEM_INSTRUCTION)]
        ),
        response_modalities=[Modality.TEXT],
    )


def create_task_config():
    return GenerateContentConfig(
        system_instruction=Content(
            role="system", parts=[Part(text=TASK_SYSTEM_INSTRUCTION)]
        ),
        response_modalities=[Modality.TEXT],
        response_mime_type="application/json",
        response_schema=TaskOrNone,
    )


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


async def generate_task(messages: List[Content]) -> TaskOrNone:
    resp = await client.aio.models.generate_content(
        model=task_model, contents=messages, config=create_task_config()
    )
    return resp.parsed


async def execute_phone_call(task: Task):
    roles_lookup = {"input": task.business_name, "output": "Bubba"}
    last_role: str | None = None
    async for transcript in make_phone_call(task):
        assert transcript.role in roles_lookup
        if last_role == transcript.role:
            resp_str = transcript.text
        else:
            last_role = transcript.role
            resp_str = f"\n\n{roles_lookup[transcript.role]}: {transcript.text}"

        yield resp_str


async def do_stream(messages: List[ClientMessage]):
    all_messages = convert_to_gemini_messages(messages)
    async with client.aio.live.connect(
        model=live_model,
        config=create_config(),
    ) as session:
        await session.send_client_content(
            turns=all_messages,
            turn_complete=True,
        )
        async for response in session.receive():
            if response.server_content.turn_complete:
                yield create_end_response(
                    finish_reason="stop",
                    usage_metadata=response.usage_metadata,
                )
                continue
            if response.text is None:
                continue

            yield create_text_response(response.text)

    task_or_none = await generate_task(all_messages)
    if task_or_none.task is not None:
        async for transcript in execute_phone_call(task_or_none.task):
            yield create_text_response(transcript)


@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query("data")):
    response = StreamingResponse(do_stream(request.messages))

    response.headers["x-vercel-ai-data-stream"] = "v1"
    return response
