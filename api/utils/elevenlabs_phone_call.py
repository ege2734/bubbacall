import asyncio

from elevenlabs.conversational_ai.conversation import ConversationInitiationData

from api.audio_stream.elevenlabs_conversation import ElevenLabsConversation
from api.audio_stream.local_speakermic_operator import LocalSpeakerMicOperator
from api.audio_stream.mongodb_forwarder import MongoDBForwarder
from api.audio_stream.stream_mediator import StreamMediator
from api.utils.mongodb import MongoDB
from api.utils.task import Task


async def stream_call(mongodb_client: MongoDB, task: Task, task_id: str):
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
                MongoDBForwarder(task_id, mongodb_client),
            ]
        )
        await new_stream_mediator.run()
    except asyncio.CancelledError:
        pass
