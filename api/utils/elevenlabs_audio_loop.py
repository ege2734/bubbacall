import asyncio

from api.audio_stream.elevenlabs_conversation import ElevenLabsConversation
from api.audio_stream.local_speakermic_operator import LocalSpeakerMicOperator
from api.audio_stream.stream_mediator import StreamMediator


async def run_new_stream_mediator():
    new_stream_mediator = StreamMediator(
        [
            LocalSpeakerMicOperator(),
            ElevenLabsConversation(),
        ]
    )
    await new_stream_mediator.run()


if __name__ == "__main__":
    asyncio.run(run_new_stream_mediator())
