from typing import override

from api.audio_stream.stream_operator import StreamOperator
from api.utils.mongodb import MongoDB
from api.utils.task import TaskStatus


class MongoDBForwarder(StreamOperator):
    """
    `Send` causes any relevant update to be sent to MongoDB task collection. It is
    appended to the updates array field. Relevant updates are:
    - input_transcription
    - output_transcription
    - output_transcription_correction
    - status
    `Receive` is a noop.
    """

    def __init__(self, task_id: str, mongodb_client: MongoDB):
        super().__init__("mongodb_forwarder")
        self.task_id = task_id
        self.mongodb_client = mongodb_client

    @override
    async def initialize(self):
        await self.mongodb_client.update_task_progress(
            task_id=self.task_id, task_status=TaskStatus.IN_PROGRESS
        )
        assert (
            await self.mongodb_client.get_task(self.task_id) is not None
        ), "Task not found"

    @override
    async def send_task(self):
        while not self.stop_event.is_set():
            stream_data = await self.get_from_send_queue()
            if stream_data is None:
                continue

            if stream_data.input_transcription:
                await self.mongodb_client.update_task_progress(
                    task_id=self.task_id,
                    message={
                        "type": "input_transcript",
                        "value": stream_data.input_transcription,
                    },
                )
            if stream_data.output_transcription:
                await self.mongodb_client.update_task_progress(
                    task_id=self.task_id,
                    message={
                        "type": "output_transcript",
                        "value": stream_data.output_transcription,
                    },
                )
            if stream_data.output_transcription_correction:
                await self.mongodb_client.update_task_progress(
                    task_id=self.task_id,
                    message={
                        "type": "output_transcript_correction",
                        "value": stream_data.output_transcription_correction,
                    },
                )

    @override
    async def receive_task(self):
        pass

    @override
    async def close(self):
        await self.mongodb_client.update_task_progress(
            task_id=self.task_id, task_status=TaskStatus.FINISHED
        )
