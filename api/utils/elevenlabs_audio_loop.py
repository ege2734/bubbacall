import asyncio
import logging

from elevenlabs.conversational_ai.conversation import ConversationInitiationData

from api.audio_stream.elevenlabs_conversation import ElevenLabsConversation
from api.audio_stream.local_speakermic_operator import LocalSpeakerMicOperator
from api.audio_stream.stream_mediator import StreamMediator


async def run_new_stream_mediator():
    new_stream_mediator = StreamMediator(
        [
            LocalSpeakerMicOperator(out_queue_max_size=100),
            ElevenLabsConversation(
                conversation_config=ConversationInitiationData(
                    dynamic_variables={
                        "task": "If my suit is ready for pickup?",
                        "business_name": "J's cleaners",
                    },
                )
            ),
        ]
    )
    await new_stream_mediator.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_new_stream_mediator())
