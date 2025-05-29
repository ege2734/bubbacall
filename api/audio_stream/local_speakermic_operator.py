import asyncio
from dataclasses import dataclass
from typing import Any, override

import pyaudio
from google.genai.types import Blob

from api.audio_stream.stream_data import StreamData
from api.audio_stream.stream_operator import StreamOperator


@dataclass
class SpeakerMicConfig:
    send_sample_rate: int = 16000
    receive_sample_rate: int = 16000
    send_frames_per_buffer: int = 4000
    receive_frames_per_buffer: int = 1000
    channels: int = 1
    audio_format: int = pyaudio.paInt16


class LocalSpeakerMicOperator(StreamOperator):
    """
    `Send` sends audio to the local speaker.
    `Receive` receives audio from the local microphone.
    """

    speakermic_config: SpeakerMicConfig
    pya: pyaudio.PyAudio
    input_stream_read_kwargs: dict[str, Any]
    input_stream = None
    output_stream = None

    def __init__(
        self,
        config: SpeakerMicConfig = SpeakerMicConfig(),
    ):
        super().__init__("localspeakermic")
        self.speakermic_config = config
        self.pya = pyaudio.PyAudio()

        if __debug__:
            self.input_stream_read_kwargs = {"exception_on_overflow": False}
        else:
            self.input_stream_read_kwargs = {}

    @override
    async def initialize(self):
        assert self.input_stream is None, "Was this class initialized already?"
        mic_info = self.pya.get_default_input_device_info()
        self.input_stream = await asyncio.to_thread(
            self.pya.open,
            format=self.speakermic_config.audio_format,
            channels=self.speakermic_config.channels,
            rate=self.speakermic_config.send_sample_rate,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=self.speakermic_config.send_frames_per_buffer,
            start=True,
        )
        self.output_stream = await asyncio.to_thread(
            self.pya.open,
            format=self.speakermic_config.audio_format,
            channels=self.speakermic_config.channels,
            rate=self.speakermic_config.receive_sample_rate,
            output=True,
            frames_per_buffer=self.speakermic_config.receive_frames_per_buffer,
            start=True,
        )

    @override
    async def send_task(self):
        while True:
            stream_data = await self.send_queue.get()
            if stream_data.blob is None:
                continue
            await asyncio.to_thread(
                self.output_stream.write,
                stream_data.blob.data,
            )

    @override
    async def receive_task(self):
        while True:
            data = await asyncio.to_thread(
                self.input_stream.read,
                self.speakermic_config.receive_frames_per_buffer,
                **self.input_stream_read_kwargs,
            )
            stream_data = StreamData(
                originator=self.name,
                blob=Blob(data=data, mime_type="audio/pcm"),
            )
            await self.receive_queue.put(stream_data)

    @override
    async def close(self):
        if self.input_stream is not None:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None

        if self.output_stream is not None:
            self.output_stream.close()
            self.output_stream = None
