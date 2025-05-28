from dataclasses import dataclass

from google.genai.types import Blob


@dataclass
class StreamData:
    originator: str
    blob: Blob | None = None
    input_transcription: str | None = None
    output_transcription: str | None = None
    thought: str | None = None
