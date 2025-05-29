import asyncio
import logging
from dataclasses import dataclass
from typing import override

import pyaudio
from google.genai.types import Blob

from api.audio_stream.stream_data import StreamData
from api.audio_stream.stream_operator import StreamOperator


@dataclass
class SpeakerMicConfig:
    mic_sample_rate: int = 16000
    speaker_sample_rate: int = 16000
    mic_frames_per_buffer: int = 4000
    speaker_frames_per_buffer: int = 1000
    channels: int = 1
    audio_format: int = pyaudio.paInt16


class LocalSpeakerMicOperator(StreamOperator):
    """
    `Send` sends audio to the local speaker.
    `Receive` receives audio from the local microphone.
    """

    speakermic_config: SpeakerMicConfig
    pya: pyaudio.PyAudio
    input_stream = None
    output_stream = None

    def __init__(
        self,
        config: SpeakerMicConfig = SpeakerMicConfig(),
        out_queue_max_size: int = 5,
    ):
        super().__init__("localspeakermic", out_queue_max_size)
        self.speakermic_config = config
        self.pya = pyaudio.PyAudio()
        self.loop = asyncio.get_running_loop()

    @override
    async def initialize(self):
        assert self.input_stream is None, "Was this class initialized already?"
        mic_info = self.pya.get_default_input_device_info()
        # Input == receive == mic
        self.input_stream = await asyncio.to_thread(
            self.pya.open,
            format=self.speakermic_config.audio_format,
            channels=self.speakermic_config.channels,
            rate=self.speakermic_config.mic_sample_rate,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=self.speakermic_config.mic_frames_per_buffer,
            stream_callback=self._input_stream_callback,
        )
        # Output == send == speaker
        self.output_stream = await asyncio.to_thread(
            self.pya.open,
            format=self.speakermic_config.audio_format,
            channels=self.speakermic_config.channels,
            rate=self.speakermic_config.speaker_sample_rate,
            output=True,
            frames_per_buffer=self.speakermic_config.speaker_frames_per_buffer,
        )

    @override
    async def send_task(self):
        while not self.stop_event.is_set():
            stream_data = await self.get_from_send_queue()
            if stream_data is None or stream_data.blob is None:
                continue
            await asyncio.to_thread(
                self.output_stream.write,
                stream_data.blob.data,
            )

    @override
    async def receive_task(self):
        pass

    @override
    async def close(self):
        logging.info("Closing local speakermic operator")
        self.input_stream.stop_stream()
        self.input_stream.close()
        self.output_stream.close()

    def _input_stream_callback(self, in_data, _frame_count, _time_info, _status):
        if self.stop_event.is_set():
            return (None, pyaudio.paComplete)
        stream_data = StreamData(
            originator=self.name,
            blob=Blob(data=in_data, mime_type="audio/pcm"),
        )
        self.loop.call_soon_threadsafe(self.receive_queue.put_nowait, stream_data)
        return (None, pyaudio.paContinue)
