import json
from typing import List

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
from twilio.rest import Client as TwilioClient

from api.utils.chat_base import BASE_INSTRUCTIONS
from api.utils.elevenlabs_phone_call import stream_call as stream_elevenlabs_call
from api.utils.mongodb import MongoDB, TaskStatus
from api.utils.prompt import ClientMessage, convert_to_gemini_messages
from api.utils.task import generate_task
from api.utils.twilio_phone_call import request_outbound_call

SYSTEM_INSTRUCTION = f"""
{BASE_INSTRUCTIONS}

If they accept, let the user know you're taking on the task and will let them know when you're done.
"""


def create_config(tools: ToolListUnion):
    return GenerateContentConfig(
        system_instruction=Content(
            role="system", parts=[Part(text=SYSTEM_INSTRUCTION)]
        ),
        response_modalities=[Modality.TEXT],
        tools=tools,
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


async def do_stream(
    gemini_client: genai.Client,
    mcp_session_group: ClientSessionGroup,
    twilio_client: TwilioClient,
    mongodb_client: MongoDB,
    messages: List[ClientMessage],
    *,
    model: str = "gemini-2.0-flash",
    fake_phone_call: bool = False,
):
    all_messages = convert_to_gemini_messages(messages)

    async for response in await gemini_client.aio.models.generate_content_stream(
        model=model,
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

    task_or_none = await generate_task(client=gemini_client, messages=all_messages)
    if task_or_none.task is not None:
        # Store the task in MongoDB
        task_id = await mongodb_client.store_task(task_or_none.task)
        if fake_phone_call:
            # Kicks off a "fake" phone call via your computer's speakermic.
            async for transcript in stream_elevenlabs_call(task_or_none.task):
                yield create_text_response(transcript)
        else:
            # Kicks off a Twilio phone call
            await request_outbound_call(task_id, twilio_client)

            # Watch and yield task updates
            async for update in mongodb_client.watch_task_updates(task_id):
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
