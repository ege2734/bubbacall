import asyncio

from elevenlabs.conversational_ai.conversation import ConversationInitiationData
from fastapi import WebSocket

from api.audio_stream.elevenlabs_conversation import ElevenLabsConversation
from api.audio_stream.stream_mediator import StreamMediator
from api.audio_stream.transcript_forwarder import TranscriptData, TranscriptForwarder
from api.audio_stream.twilio_call import TwilioCall
from api.utils.elevenlabs_phone_call import Task


async def make_phone_call(twilio_websocket: WebSocket, task: Task, task_id: str):
    transcript_queue: asyncio.Queue[TranscriptData] = asyncio.Queue()
    try:
        new_stream_mediator = StreamMediator(
            [
                TwilioCall(twilio_websocket),
                ElevenLabsConversation(
                    conversation_config=ConversationInitiationData(
                        dynamic_variables={
                            "task": task.task,
                            "business_name": task.business_name,
                        },
                    )
                ),
                # TODO(ege): Replace this with "DatabaseForwarder" that just writes
                # the updates to MongoDb directly.
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
