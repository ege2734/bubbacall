from typing import override

from google import genai
from google.genai.types import Blob

from api.audio_stream.stream_data import StreamData
from api.audio_stream.stream_operator import StreamOperator


class GeminiStreamOperator(StreamOperator):
    """
    `Send` sends audio to the Gemini model.
    `Receive` receives audio from the Gemini model. It also populates the
    `input_transcription` and `output_transcription` fields if those are returned
    by the Gemini model.

    This operator does not own the lifetime of the `session`.
    """

    def __init__(self, session: genai.live.AsyncSession):
        super().__init__("gemini_stream")
        self.session = session

    @override
    async def send_task(self):
        while True:
            stream_data = await self.send_queue.get()
            if stream_data.blob is None:
                return
            await self.session.send_realtime_input(audio=stream_data.blob)

    @override
    async def receive_task(self):
        while True:
            turn = self.session.receive()
            async for response in turn:
                blob: Blob | None = None
                input_transcription: str | None = None
                output_transcription: str | None = None
                if response.server_content.input_transcription:
                    input_transcription = (
                        response.server_content.input_transcription.text
                    )
                    print(f"Input transcription: {input_transcription}")
                if response.server_content.output_transcription:
                    output_transcription = (
                        response.server_content.output_transcription.text
                    )
                    print(f"Output transcription: {output_transcription}")
                if data := response.data:
                    blob = Blob(data=data, mime_type="audio/pcm")

                stream_data = StreamData(
                    originator=self.name,
                    blob=blob,
                    input_transcription=input_transcription,
                    output_transcription=output_transcription,
                )
                await self.receive_queue.put(stream_data)

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.receive_queue.empty():
                self.receive_queue.get_nowait()
