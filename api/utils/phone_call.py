import asyncio
import logging

from google import genai
from google.genai import types
from google.genai.types import Content, Part
from pydantic import BaseModel

from api.audio_stream.gemini_stream_operator import GeminiStreamOperator
from api.audio_stream.local_speakermic_operator import LocalSpeakerMicOperator
from api.audio_stream.stream_mediator import StreamMediator
from api.audio_stream.transcript_forwarder import TranscriptData, TranscriptForwarder

from .settings import get_setting


class Task(BaseModel):
    business_name: str
    business_phone_number: str
    task: str


HANGUP_MAGIC_PHRASE = "FizzBuzz this phone call has ended"


def get_system_instruction(task: Task) -> str:
    return f"""
    You are a helpful AI phone assistant. You are connected to another human via a phone call.
    The person on the other end of the phone call should be somebody who is picking up the phone
    for a business {task.business_name}. You are tasked with completing the following task:
    {task.task}.
    
    You should fully assume the person on the other side thinks you are a human and act that way.
    You should not reveal that you are an AI.
    
    Navigate the phone call to complete the task.
    
    Throughout the conversation, do not indulge in any conversation that is not related to the task.
    Do not reveal that you are an AI.
    Do not reveal that you are a computer program.
    """


def make_config(system_instruction: str) -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        enable_affective_dialog=True,
        proactivity={"proactive_audio": True},
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=Content(
            role="system", parts=[Part(text=system_instruction)]
        ),
        # tools=[
        #     types.Tool(
        #         function_declarations=[
        #             types.FunctionDeclaration(
        #                 name="end_call",
        #                 description="Call this function when you have completed your task and the call should be ended",
        #                 parameters=types.Schema(),
        #             )
        #         ]
        #     )
        # ],
    )


client = genai.Client(
    api_key=get_setting("GEMINI_API_KEY"), http_options={"api_version": "v1alpha"}
)
# The thinking one: gemini-2.5-flash-exp-native-audio-thinking-dialog
MODEL = "gemini-2.5-flash-preview-native-audio-dialog"


async def make_phone_call(task: Task):
    transcript_queue: asyncio.Queue[TranscriptData] = asyncio.Queue()
    try:
        async with client.aio.live.connect(
            model=MODEL, config=make_config(get_system_instruction(task))
        ) as session:

            new_stream_mediator = StreamMediator(
                [
                    LocalSpeakerMicOperator(),
                    GeminiStreamOperator(session=session),
                    TranscriptForwarder(out_queue=transcript_queue),
                ]
            )
            call_task = asyncio.create_task(new_stream_mediator.run())
            get_queue_task = asyncio.create_task(transcript_queue.get())
            while not call_task.done():
                await asyncio.wait(
                    [call_task, get_queue_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if get_queue_task.done():
                    yield get_queue_task.result()
                    get_queue_task = asyncio.create_task(transcript_queue.get())
    except asyncio.CancelledError:
        pass
    finally:
        call_task.cancel()
        get_queue_task.cancel()
