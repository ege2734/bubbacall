from typing import List

from google import genai
from google.genai.types import Content, GenerateContentConfig, Modality, Part
from pydantic import BaseModel

from api.utils.chat_base import BASE_INSTRUCTIONS

TASK_SYSTEM_INSTRUCTION = f"""
{BASE_INSTRUCTIONS}

If they accept, your next message should be in the json format provided alongside your config.

Your response should have .task set to 'None' UNTIL the user confirms the task
"""


class Task(BaseModel):
    business_name: str
    business_phone_number: str
    task: str


class TaskOrNone(BaseModel):
    task: Task | None = None


def create_task_config():
    return GenerateContentConfig(
        system_instruction=Content(
            role="system", parts=[Part(text=TASK_SYSTEM_INSTRUCTION)]
        ),
        response_modalities=[Modality.TEXT],
        response_mime_type="application/json",
        response_schema=TaskOrNone,
    )


async def generate_task(
    client: genai.Client, messages: List[Content], model: str = "gemini-2.0-flash"
) -> TaskOrNone:
    resp = await client.aio.models.generate_content(
        model=model, contents=messages, config=create_task_config()
    )
    return resp.parsed
