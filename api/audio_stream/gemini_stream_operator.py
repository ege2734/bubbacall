from typing import override

from google import genai
from google.genai.types import Blob, LiveServerMessage

from api.audio_stream.stream_data import StreamData
from api.audio_stream.stream_operator import StreamOperator


def _get_data(resp: LiveServerMessage) -> bytes | None:
    if (
        not resp.server_content
        or not resp.server_content
        or not resp.server_content.model_turn
        or not resp.server_content.model_turn.parts
    ):
        return None
    concatenated_data = b""
    for part in resp.server_content.model_turn.parts:
        if part.inline_data and isinstance(part.inline_data.data, bytes):
            concatenated_data += part.inline_data.data

    return concatenated_data if len(concatenated_data) > 0 else None


def _get_thought(resp: LiveServerMessage) -> str | None:
    if (
        not resp.server_content
        or not resp.server_content
        or not resp.server_content.model_turn
        or not resp.server_content.model_turn.parts
    ):
        return None
    text = ""
    for part in resp.server_content.model_turn.parts:
        if part.text is not None:
            if part.thought is not None and part.thought:
                text += part.text

    return text if text else None


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
        while not self.stop_event.is_set():
            stream_data = await self.send_queue.get()
            if stream_data.blob is None:
                continue
            await self.session.send_realtime_input(audio=stream_data.blob)

    @override
    async def receive_task(self):
        while not self.stop_event.is_set():
            turn = self.session.receive()
            async for response in turn:
                blob: Blob | None = None
                input_transcription: str | None = None
                output_transcription: str | None = None
                thought: str | None = None
                if response.server_content.input_transcription:
                    input_transcription = (
                        response.server_content.input_transcription.text
                    )
                if response.server_content.output_transcription:
                    output_transcription = (
                        response.server_content.output_transcription.text
                    )
                if data := response.data:
                    blob = Blob(data=data, mime_type="audio/pcm")
                thought = _get_thought(response)

                stream_data = StreamData(
                    originator=self.name,
                    blob=blob,
                    input_transcription=input_transcription,
                    output_transcription=output_transcription,
                    thought=thought,
                )
                await self.receive_queue.put(stream_data)

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.receive_queue.empty() and not self.stop_event.is_set():
                self.receive_queue.get_nowait()
