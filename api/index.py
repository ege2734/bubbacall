import json
import logging
from typing import List

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from google import genai
from google.genai.types import Content, LiveConnectConfig, Modality, Part, UsageMetadata
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
SYSTEM_INSTRUCTION = f"""
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

If they accept, your next message should be:
{BEGIN_TASK_DEFINITION}
{{"business_name": {{business_name}}, "business_phone_number": {{business_phone_number}}, "task": {{task}}}}
{END_TASK_DEFINITION}

It is a grave error to not follow the above instructions. It is an even worse error
to not abide by the output format given to you below. This format will be parsed via
python, therefore it is imperative that you follow the format exactly.
"""

model = "gemini-2.0-flash-live-001"
config = LiveConnectConfig(
    system_instruction=Content(role="system", parts=[Part(text=SYSTEM_INSTRUCTION)]),
    response_modalities=[Modality.TEXT],
)

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
    all_messages = convert_to_gemini_messages(messages)
    async with client.aio.live.connect(
        model=model,
        config=config,
    ) as session:
        await session.send_client_content(
            turns=all_messages,
            turn_complete=True,
        )
        task_def: str | None = None
        async for response in session.receive():
            if response.server_content.turn_complete:
                yield create_end_response(
                    finish_reason="stop",
                    usage_metadata=response.usage_metadata,
                )
                continue
            if response.text is None:
                continue

            if BEGIN_TASK_DEFINITION in response.text:
                assert task_def is None

                if END_TASK_DEFINITION in response.text:
                    task_def = (
                        response.text.split(BEGIN_TASK_DEFINITION)[1]
                        .split(END_TASK_DEFINITION)[0]
                        .strip()
                    )
                    print(f"The full task appears to be: {task_def}")
                    task = json.loads(task_def)
                    await make_phone_call(Task(**task))
                else:
                    task_def = response.text.split(BEGIN_TASK_DEFINITION)[1]
                continue

            if task_def is not None:
                if END_TASK_DEFINITION in response.text:
                    task_def += response.text.split(END_TASK_DEFINITION)[0].strip()
                    print(f"The task appears to be: {task_def}")
                    task = json.loads(task_def)
                    await make_phone_call(Task(**task))
                    continue
                task_def += response.text
            else:
                yield create_text_response(response.text)


@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query("data")):
    response = StreamingResponse(do_stream(request.messages))

    response.headers["x-vercel-ai-data-stream"] = "v1"
    return response
