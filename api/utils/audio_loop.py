import asyncio

from google import genai
from google.genai import types

from api.audio_stream.gemini_stream_operator import GeminiStreamOperator
from api.audio_stream.local_speakermic_operator import LocalSpeakerMicOperator
from api.audio_stream.stream_mediator import StreamMediator
from api.utils.settings import get_setting

# GOOGLE_API_KEY must be set as env variable
client = genai.Client(
    api_key=get_setting("GEMINI_API_KEY"), http_options={"api_version": "v1alpha"}
)

# The thinking one: gemini-2.5-flash-exp-native-audio-thinking-dialog
MODEL = "gemini-2.5-flash-preview-native-audio-dialog"
CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    enable_affective_dialog=True,
    proactivity={"proactive_audio": True},
    # realtime_input_config=types.RealtimeInputConfig(
    #     activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    # ),
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
)


async def run_new_stream_mediator():
    async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
        new_stream_mediator = StreamMediator(
            [
                LocalSpeakerMicOperator(),
                GeminiStreamOperator(session=session),
            ]
        )
        await new_stream_mediator.run()


if __name__ == "__main__":
    asyncio.run(run_new_stream_mediator())
    # loop = AudioLoop()
    # asyncio.run(loop.run())
