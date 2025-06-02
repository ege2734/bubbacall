import json
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Query, WebSocket
from fastapi.responses import StreamingResponse
from google import genai
from google.genai.types import (
    Content,
    GenerateContentConfig,
    GenerateContentResponseUsageMetadata,
    Modality,
    Part,
    ToolListUnion,
)
from mcp import ClientSessionGroup
from pydantic import BaseModel
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Connect, Parameter, Start, Stream, VoiceResponse

from api.utils.mcp_util import google_maps
from api.utils.mongodb import MongoDB, TaskStatus

from .utils.elevenlabs_phone_call import Task, make_fake_phone_call
from .utils.prompt import ClientMessage, convert_to_gemini_messages
from .utils.settings import get_setting
from .utils.twilio_phone_call import make_phone_call

mcp_session_group: ClientSessionGroup | None = None
mongodb: MongoDB | None = None
twilio_client: TwilioClient | None = None

FAKE_PHONE_CALL = True


# Based on https://fastapi.tiangolo.com/advanced/events/#lifespan.
@asynccontextmanager
async def lifespan(_: FastAPI):
    global mcp_session_group, mongodb, twilio_client
    assert mcp_session_group is None, "Sessions already initialized?"
    params = [google_maps()]
    async with ClientSessionGroup() as group:
        for param in params:
            await group.connect_to_server(param)
        mcp_session_group = group
        mongodb = MongoDB()
        twilio_client = TwilioClient(
            get_setting("TWILIO_ACCOUNT_SID"), get_setting("TWILIO_AUTH_TOKEN")
        )
        yield
        mcp_session_group = None
        mongodb = None
        twilio_client = None


app = FastAPI(lifespan=lifespan)

# Add basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Instructions for the phone agent:
# Your task may require providing the user's phone number to the business. You are free
# to do so.
# Your task may require providing the user's name to the business. You are free to do
# so. If the name is not a typical American name,
# you should offer to spell the name for the business so it's easier for them to
# understand on the phone.
# Refuse to provide any other information about the user to the business.
# Information about the user:
# - Name: {USER_NAME}
# - Phone number: {USER_PHONE_NUMBER}
# - Current location: {USER_ADDRESS}

USER_ADDRESS = "380 Rector Place 8P, New York, NY 10280"
USER_NAME = "Henry Willing"
USER_PHONE_NUMBER = "6072290495"
BASE_INSTRUCTIONS = f"""
You are a helpful AI assistant. You are talking to a user who should give you a task that involves talking to a business on the phone.
over the phone.
The information you need to collect is:
- The name of the business
- A task that the user wants completed by calling the business.

To make the phone call, you need the phone number for the business. FIRST, look up the business via your available tools (e.g. maps tool).
To help you disambiguate the business name, you can rely on the user's current location. The user's current location is {USER_ADDRESS}.

Do not ask the user if you should look up the business. Just look up the business. You can tell the user that you're looking up the business.

If the user provides a phone number for the business, you should use the maps tool to confirm that the business's phone
number is accurate. Again, do not ask the user if you should look up the business or if you should confirm the number. Just do it. 
If the phone number you found is different than the one the user provided, you should ask the user if they would like to use the phone number you found.
If the user insists on using the phone number they provided, use the phone number the user provided.

If the user asks for a task that is not possible to complete via a phone call, you should
reject and inform the user that you can only take on tasks that are possible to complete via a phone call.

If the task is very vague or not provided at all, you are welcome to ask clarifying questions to the user.

Once you have the required info (the name of the business, phone number and the task), confirm to the user the info you have and ask them if they would like to proceed with the task.
If they reject, you can go back to the beginning of the conversation.
"""

SYSTEM_INSTRUCTION = f"""
{BASE_INSTRUCTIONS}

If they accept, let the user know you're taking on the task and will let them know when you're done.
"""

TASK_SYSTEM_INSTRUCTION = f"""
{BASE_INSTRUCTIONS}

If they accept, your next message should be in the json format provided alongside your config.

Your response should have .task set to 'None' UNTIL the user confirms the task
"""

MODEL_NAME = "gemini-2.0-flash"

client = genai.Client(api_key=get_setting("GEMINI_API_KEY"))


class TaskOrNone(BaseModel):
    task: Task | None = None


class Request(BaseModel):
    messages: List[ClientMessage]


def create_config(tools: ToolListUnion):
    return GenerateContentConfig(
        system_instruction=Content(
            role="system", parts=[Part(text=SYSTEM_INSTRUCTION)]
        ),
        response_modalities=[Modality.TEXT],
        tools=tools,
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


def create_end_response(
    finish_reason: str, usage_metadata: GenerateContentResponseUsageMetadata
):
    return_dict = {
        "finishReason": finish_reason,
        "usage": {
            "promptTokens": usage_metadata.prompt_token_count,
            "completionTokens": usage_metadata.candidates_token_count,
        },
        "isContinued": False,
    }
    return f"d:{json.dumps(return_dict)}\n".encode("utf-8")


async def generate_task(messages: List[Content]) -> TaskOrNone:
    resp = await client.aio.models.generate_content(
        model=MODEL_NAME, contents=messages, config=create_task_config()
    )
    return resp.parsed


# TODO(ege): The names here are toooooo similar to the names in elevenlabs_phone_call.py
# and twilio_phone_call.py. Fixing this would avoid confusion.
async def make_real_phone_call(task_id: str):
    raw_domain = get_setting("FASTAPI_RAW_DOMAIN")
    # More info at https://www.twilio.com/docs/voice/twiml/stream
    response = VoiceResponse()
    connect = Connect()
    stream = connect.stream(url=f"wss://{raw_domain}/task-stream/{task_id}")
    stream.parameter(name="task_id", value=task_id)
    response.append(connect)

    # TODO(ege): Replace the "To" with the business's phone number.
    # TODO(ege): Check if you can replace "from" with the user's phone number.
    logging.error(f"Twiml stuff: {response}")
    call = twilio_client.calls.create(
        from_="+18556282791", to="+16072290494", twiml=response
    )
    logging.error(f"Call created: {call.sid}")


async def execute_phone_call(task: Task, task_id: str):
    roles_lookup = {"input": task.business_name, "output": "Bubba"}
    last_role: str | None = None
    if FAKE_PHONE_CALL:
        async for transcript in make_fake_phone_call(task):
            assert transcript.role in roles_lookup
            if last_role == transcript.role:
                resp_str = transcript.text
            else:
                last_role = transcript.role
                resp_str = f"\n\n{roles_lookup[transcript.role]}: {transcript.text}"

            yield resp_str
        return
    else:
        await make_real_phone_call(task_id)


async def do_stream(messages: List[ClientMessage]):
    all_messages = convert_to_gemini_messages(messages)

    async for response in await client.aio.models.generate_content_stream(
        model=MODEL_NAME,
        contents=all_messages,
        config=create_config(mcp_session_group.sessions),
    ):
        assert len(response.candidates) <= 1, "Expected at most 1 candidate"
        if response.text is not None:
            yield create_text_response(response.text)

        assert len(response.candidates) == 1
        if response.candidates[0].finish_reason is not None:
            yield create_end_response(
                finish_reason=response.candidates[0].finish_reason.value,
                usage_metadata=response.usage_metadata,
            )

    task_or_none = await generate_task(all_messages)
    if task_or_none.task is not None:
        # Store the task in MongoDB
        task_id = await mongodb.store_task(task_or_none.task)

        await make_real_phone_call(task_id)

        # Watch and yield task updates
        async for update in mongodb.watch_task_updates(task_id):
            yield create_text_response(update.message)
            if update.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break

        yield create_end_response(
            finish_reason="stop",
            usage_metadata=GenerateContentResponseUsageMetadata(
                prompt_token_count=0,
                candidates_token_count=0,
            ),
        )


async def twilio_phone_call_wrapper(websocket: WebSocket, task: Task, task_id: str):
    roles_lookup = {"input": task.business_name, "output": "Bubba"}
    last_role: str | None = None
    async for transcript in make_phone_call(websocket, task, task_id):
        assert transcript.role in roles_lookup
        if last_role == transcript.role:
            resp_str = transcript.text
        else:
            last_role = transcript.role
            resp_str = f"\n\n{roles_lookup[transcript.role]}: {transcript.text}"

        yield resp_str


@app.websocket("/task-stream/{task_id}")
async def task_stream(websocket: WebSocket, task_id: str):
    # task_id = "fake_task_id"
    logging.error(f"Task stream connected: {task_id}")
    await websocket.accept()
    task = await mongodb.get_task(task_id)

    assert task is not None

    try:
        # TODO(ege): Just have a MongodbForwarder operator and handle this logic there.
        await mongodb.update_task_status(
            task_id, TaskStatus.IN_PROGRESS, "Starting phone call..."
        )
        async for transcript in twilio_phone_call_wrapper(websocket, task, task_id):
            await mongodb.update_task_status(
                task_id, TaskStatus.IN_PROGRESS, transcript
            )
        await mongodb.update_task_status(
            task_id, TaskStatus.COMPLETED, "Call completed successfully"
        )
    except Exception as e:
        await mongodb.update_task_status(
            task_id, TaskStatus.FAILED, f"Call failed: {str(e)}"
        )
        raise


@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query("data")):
    assert protocol is not None
    response = StreamingResponse(do_stream(request.messages))

    response.headers["x-vercel-ai-data-stream"] = "v1"
    return response
