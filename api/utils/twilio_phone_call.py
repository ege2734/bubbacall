import asyncio
import logging

from elevenlabs.conversational_ai.conversation import ConversationInitiationData
from fastapi import WebSocket
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Connect, VoiceResponse

from api.audio_stream.elevenlabs_conversation import ElevenLabsConversation
from api.audio_stream.stream_mediator import StreamMediator
from api.audio_stream.transcript_forwarder import TranscriptData, TranscriptForwarder
from api.audio_stream.twilio_call import TwilioCall
from api.utils.mongodb import MongoDB, TaskStatus
from api.utils.settings import get_setting
from api.utils.task import Task


async def _stream_call(twilio_websocket: WebSocket, task: Task, task_id: str):
    transcript_queue: asyncio.Queue[TranscriptData] = asyncio.Queue()
    logging.info(f"Starting stream for twilio call for task {task_id}")
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


async def stream_call(mongodb: MongoDB, websocket: WebSocket, task: Task, task_id: str):
    roles_lookup = {"input": task.business_name, "output": "Bubba"}
    last_role: str | None = None
    await mongodb.update_task_status(
        task_id, TaskStatus.IN_PROGRESS, "Starting phone call..."
    )
    try:
        # TODO(ege): Just have a MongodbForwarder operator and handle this logic there.
        async for transcript in _stream_call(websocket, task, task_id):
            assert transcript.role in roles_lookup
            if last_role == transcript.role:
                resp_str = transcript.text
            else:
                last_role = transcript.role
                resp_str = f"\n\n{roles_lookup[transcript.role]}: {transcript.text}"

            await mongodb.update_task_status(task_id, TaskStatus.IN_PROGRESS, resp_str)
        await mongodb.update_task_status(
            task_id, TaskStatus.COMPLETED, "Call completed successfully"
        )
    except Exception as e:
        await mongodb.update_task_status(
            task_id, TaskStatus.FAILED, f"Call failed: {str(e)}"
        )
        raise


async def request_outbound_call(task_id: str, twilio_client: TwilioClient):
    """
    Makes an HTTP request to start an outbound call. Loosely based on the example
    in https://github.com/twilio/media-streams.
    Additional references for the TWiML stuff (as well as handling the communication
    lifetime with Twilio):
       1. https://www.twilio.com/docs/voice/twiml/stream.
       2. Call resource: https://www.twilio.com/docs/voice/api/call-resource#create-a-call-resource
    """
    raw_domain = get_setting("FASTAPI_RAW_DOMAIN")
    response = VoiceResponse()
    connect = Connect()
    # At some point, task_id may need to be passed as a custom parameter.
    # https://www.twilio.com/docs/voice/twiml/stream#custom-parameters
    stream = connect.stream(url=f"wss://{raw_domain}/task-stream/{task_id}")
    stream.parameter(name="task_id", value=task_id)
    response.append(connect)

    # TODO(ege): Replace the "To" with the business's phone number.
    # TODO(ege): Check if you can replace "from" with the user's phone number.
    logging.info(f"Twiml stuff: {response}")
    call = twilio_client.calls.create(
        from_="+18556282791", to="+16072290494", twiml=response
    )
    logging.info(f"Call created: {call.sid}")
