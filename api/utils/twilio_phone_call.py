import asyncio
import logging

from elevenlabs.conversational_ai.conversation import ConversationInitiationData
from fastapi import WebSocket
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Connect, VoiceResponse

from api.audio_stream.elevenlabs_conversation import ElevenLabsConversation
from api.audio_stream.mongodb_forwarder import MongoDBForwarder
from api.audio_stream.stream_mediator import StreamMediator
from api.audio_stream.twilio_call import TwilioCall
from api.utils.mongodb import MongoDB
from api.utils.settings import get_setting
from api.utils.task import Task


async def stream_call(
    mongodb_client: MongoDB, websocket: WebSocket, task: Task, task_id: str
):
    logging.info(f"Starting stream for twilio call for task {task_id}")
    try:
        new_stream_mediator = StreamMediator(
            [
                TwilioCall(websocket),
                ElevenLabsConversation(
                    conversation_config=ConversationInitiationData(
                        dynamic_variables={
                            "task": task.task,
                            "business_name": task.business_name,
                        },
                    )
                ),
                MongoDBForwarder(task_id, mongodb_client),
            ]
        )
        await new_stream_mediator.run()
    except asyncio.CancelledError:
        pass


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
