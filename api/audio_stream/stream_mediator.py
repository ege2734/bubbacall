import asyncio
import traceback

import aiostream

from api.audio_stream.stream_operator import StreamOperator


class StreamMediator:
    # Take in a list of "operator" classes, which implement a standard
    # send and receive interfaces.
    #
    def __init__(self, operators: list[StreamOperator]):
        """Initialize with a list of stream operators that will send/receive data.

        Args:
            operators: List of StreamOperator instances to mediate between
        """
        self.operators = operators
        self.tasks: list[asyncio.Task] = []

    async def run(self):
        """Run all operator tasks and handle message routing between them."""
        # Start all send and receive tasks
        try:
            async with asyncio.TaskGroup() as tg:
                for op in self.operators:
                    self.tasks.append(tg.create_task(op.send_task()))
                    self.tasks.append(tg.create_task(op.receive_task()))

                # Route messages between operators
                while True:
                    combined_stream = aiostream.stream.merge(
                        *[op.receive() for op in self.operators]
                    )
                    async with combined_stream.stream() as receive_stream:
                        async for stream_data in receive_stream:
                            # Forward received data to all other operators
                            for op in self.operators:
                                await op.send(stream_data)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            for op in self.operators:
                op.close()
            traceback.print_exception(e)
