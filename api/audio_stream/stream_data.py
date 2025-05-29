from dataclasses import dataclass

from google.genai.types import Blob


@dataclass
class TranscriptCorrection:
    original: str
    corrected: str


@dataclass
class StreamData:
    originator: str
    # TODO: Define our own blob and use it here.
    blob: Blob | None = None
    # TODO: Prolly call these "transcript"
    input_transcription: str | None = None
    output_transcription: str | None = None
    output_transcription_correction: TranscriptCorrection | None = None
    # TODO: Thought is not used rn, prolly could be removed.
    thought: str | None = None

    # TODO: This needs to be wired.
    force_end_call: bool = False
