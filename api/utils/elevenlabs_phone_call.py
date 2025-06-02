import asyncio

from elevenlabs.conversational_ai.conversation import ConversationInitiationData
from pydantic import BaseModel

from api.audio_stream.elevenlabs_conversation import ElevenLabsConversation
from api.audio_stream.local_speakermic_operator import LocalSpeakerMicOperator
from api.audio_stream.stream_mediator import StreamMediator
from api.audio_stream.transcript_forwarder import TranscriptData, TranscriptForwarder


class Task(BaseModel):
    business_name: str
    business_phone_number: str
    task: str


async def make_fake_phone_call(task: Task):
    transcript_queue: asyncio.Queue[TranscriptData] = asyncio.Queue()
    try:
        new_stream_mediator = StreamMediator(
            [
                LocalSpeakerMicOperator(out_queue_max_size=100),
                ElevenLabsConversation(
                    conversation_config=ConversationInitiationData(
                        dynamic_variables={
                            "task": task.task,
                            "business_name": task.business_name,
                        },
                    )
                ),
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
