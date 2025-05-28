import asyncio
import traceback
from dataclasses import dataclass

import pyaudio
from google import genai
from google.genai import types
from google.genai.types import Content, Part
from pydantic import BaseModel

from api.audio_stream.gemini_stream_operator import GeminiStreamOperator
from api.audio_stream.local_speakermic_operator import LocalSpeakerMicOperator
from api.audio_stream.stream_mediator import StreamMediator
from api.audio_stream.transcript_forwarder import TranscriptData, TranscriptForwarder

from .settings import get_setting

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

pya = pyaudio.PyAudio()


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
    
    Once you have completed the task, you should thank the user. You don't have the capability to hang up, so instead
    you should say "{HANGUP_MAGIC_PHRASE}".
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
    )


client = genai.Client(
    api_key=get_setting("GEMINI_API_KEY"), http_options={"api_version": "v1alpha"}
)
# The thinking one: gemini-2.5-flash-exp-native-audio-thinking-dialog
MODEL = "gemini-2.5-flash-preview-native-audio-dialog"


class AudioLoop:
    def __init__(self, config: types.LiveConnectConfig):
        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.audio_stream = None

        self.receive_audio_task = None
        self.play_audio_task = None
        self.config = config

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
                client.aio.live.connect(model=MODEL, config=self.config) as session,
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


async def make_phone_call(task: Task):
    transcript_queue: asyncio.Queue[TranscriptData] = asyncio.Queue()
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
        while not call_task.done():
            get_queue_task = asyncio.create_task(transcript_queue.get())
            await asyncio.wait(
                [call_task, get_queue_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if get_queue_task.done():
                yield get_queue_task.result()
