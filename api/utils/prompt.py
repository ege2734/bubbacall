from enum import Enum
from typing import Any, List, Optional

from google.genai.types import Content, Part
from pydantic import BaseModel

from .attachment import ClientAttachment


class ToolInvocationState(str, Enum):
    CALL = "call"
    PARTIAL_CALL = "partial-call"
    RESULT = "result"


class ToolInvocation(BaseModel):
    state: ToolInvocationState
    toolCallId: str
    toolName: str
    args: Any
    result: Any


class ClientMessage(BaseModel):
    role: str
    content: str
    experimental_attachments: Optional[List[ClientAttachment]] = None
    toolInvocations: Optional[List[ToolInvocation]] = None


def convert_to_gemini_messages(messages: List[ClientMessage]) -> list[Content]:
    gemini_messages: list[Content] = []

    for message in messages:
        gemini_messages.append(
            Content(role=message.role, parts=[Part(text=message.content)])
        )

    return gemini_messages
