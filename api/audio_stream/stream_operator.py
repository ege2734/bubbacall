import asyncio
from abc import abstractmethod
from typing import AsyncGenerator

from api.audio_stream.stream_data import StreamData


class StreamOperator:
    # Interface for defining stream operators. Example implementations can be
    # gemini audio, twilio phone api, etc.
    # Implements send and receive methods.
    # Send is an async endpoint that accepts StreamData. And "sends" it to
    # whatever agent this operator represents (e.g. phone call endpoint, gemini audio endpoint, etc)
    # Receive is an async generator (similar to session.receive()) that yields StreamData.
    # This represents the data/audio that the agent received and should
    # be forwarded to all other agents (via their send method).
    def __init__(self, name: str, out_queue_max_size: int = 5):
        self.name = name
        self.send_queue: asyncio.Queue[StreamData] = asyncio.Queue()
        self.receive_queue: asyncio.Queue[StreamData] = asyncio.Queue(
            maxsize=out_queue_max_size
        )
        self.operator_done: bool = False

    @abstractmethod
    async def send_task(self):
        pass

    async def send(self, stream_data: StreamData):
        if stream_data.originator == self.name:
            # Avoid sending data to ourselves.
            return
        await self.send_queue.put(stream_data)

    @abstractmethod
    async def receive_task(self) -> None:
        pass

    async def receive(self) -> AsyncGenerator[StreamData]:
        while True:
            data = await self.receive_queue.get()
            assert (
                data.originator == self.name
            ), "receive_task should add its own name to the stream data"
            yield data

    async def close(self) -> None:
        pass
