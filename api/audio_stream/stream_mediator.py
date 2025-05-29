import asyncio
import logging
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
        self.stop_event = asyncio.Event()

    async def run(self):
        """Run all operator tasks and handle message routing between them."""
        # Start all send and receive tasks
        try:
            async with asyncio.TaskGroup() as tg:
                for op in self.operators:
                    await op.initialize()

                for op in self.operators:
                    self.tasks.append(
                        tg.create_task(op.send_task(), name=f"{op.name}-send")
                    )
                    self.tasks.append(
                        tg.create_task(op.receive_task(), name=f"{op.name}-receive")
                    )

                # Route messages between operators
                while True:
                    combined_stream = aiostream.stream.merge(
                        *[op.receive() for op in self.operators]
                    )
                    async with combined_stream.stream() as receive_stream:
                        async for stream_data in receive_stream:
                            if stream_data.force_end_call:
                                logging.info("Detected force_end_call")
                                for op in self.operators:
                                    logging.info("Setting stop_event for %s", op.name)
                                    op.stop_event.set()
                                self.stop_event.set()
                                logging.info("Setting stop_event for self")
                                break
                            # Forward received data to all other operators
                            for op in self.operators:
                                await op.send(stream_data)

                    if self.stop_event.is_set():
                        break
                logging.info("Exiting tg block")
                logging.info(f"Tasks: {[t.get_name() for t in tg._tasks]}")
            logging.info("Exiting try block")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            traceback.print_exception(e)
        finally:
            logging.info("Into finally block")
            await asyncio.sleep(0.5)
            for op in self.operators:
                logging.info(f"Closing {op.name}")
                await op.close()
