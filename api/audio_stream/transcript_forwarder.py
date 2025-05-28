import asyncio
from dataclasses import dataclass
from typing import override

from api.audio_stream.stream_operator import StreamOperator


@dataclass
class TranscriptData:
    role: str
    text: str


class TranscriptForwarder(StreamOperator):
    """
    `Send` causes any transcription in the data to be put on the queue
    `Receive` is a noop.
    """

    def __init__(self, out_queue: asyncio.Queue[TranscriptData]):
        super().__init__("transcript_forwarder")
        self.out_queue = out_queue

    @override
    async def send_task(self):
        while True:
            stream_data = await self.send_queue.get()

            if stream_data.input_transcription:
                self.out_queue.put(
                    TranscriptData(role="input", text=stream_data.input_transcription)
                )
            if stream_data.output_transcription:
                self.out_queue.put(
                    TranscriptData(role="output", text=stream_data.output_transcription)
                )

    @override
    async def receive_task(self):
        pass
