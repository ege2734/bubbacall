import asyncio
import traceback

import pyaudio
from google import genai
from google.genai import types
from settings import get_setting

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

pya = pyaudio.PyAudio()

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


# Inspired by https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI_NativeAudio.py.
class AudioLoop:
    def __init__(self):
        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.audio_stream = None

        self.receive_audio_task = None
        self.play_audio_task = None

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send_realtime_input(audio=msg)

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if response.server_content.input_transcription:
                    print(
                        "User:",
                        response.server_content.input_transcription.text,
                    )
                if response.server_content.output_transcription:
                    print(
                        "Transcript:",
                        response.server_content.output_transcription.text,
                    )
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(e)


if __name__ == "__main__":
    loop = AudioLoop()
    asyncio.run(loop.run())
