import asyncio
import base64
import json
import logging
from typing import override

import websockets
from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import ConversationInitiationData
from google.genai.types import Blob

from api.audio_stream.stream_data import StreamData, TranscriptCorrection
from api.audio_stream.stream_operator import StreamOperator
from api.utils.settings import get_setting


class ElevenLabsConversation(StreamOperator):
    """
    `Send` sends audio to ElevenLabs.
    `Receive` receives audio from ElevenLabs. It also populates the
    `input_transcription` and `output_transcription` fields if those are returned
    by ElevenLabs

    This operator does not own the lifetime of the `client`.
    """

    def __init__(
        self,
        conversation_config: ConversationInitiationData = ConversationInitiationData(),
    ):
        super().__init__(
            "elevenlabs_conversation",
        )
        self.client = ElevenLabs(api_key=get_setting("ELEVENLABS_API_KEY"))
        self.agent_id = get_setting("ELEVENLABS_AGENT_ID")
        self.session = None
        self.conversation_config = conversation_config
        self._conversation_id = None
        self._last_interrupt_id = 0

    @override
    async def initialize(self):
        self.session = await websockets.connect(
            self._get_signed_url(), max_size=16 * 1024 * 1024
        )

        # Send initial configuration
        await self.session.send(
            json.dumps(
                {
                    "type": "conversation_initiation_client_data",
                    "custom_llm_extra_body": self.conversation_config.extra_body,
                    "conversation_config_override": self.conversation_config.conversation_config_override,
                    "dynamic_variables": self.conversation_config.dynamic_variables,
                }
            )
        )

    def _get_signed_url(self):
        response = self.client.conversational_ai.conversations.get_signed_url(
            agent_id=self.agent_id
        )
        return response.signed_url

    @override
    async def send_task(self):
        try:
            while not self.stop_event.is_set():
                stream_data = await self.get_from_send_queue()
                if stream_data is None or stream_data.blob is None:
                    continue
                await self.session.send(
                    json.dumps(
                        {
                            "user_audio_chunk": base64.b64encode(
                                stream_data.blob.data
                            ).decode()
                        }
                    )
                )
        except websockets.exceptions.ConnectionClosedOK:
            if not self.stop_event.is_set():
                await self.receive_queue.put(
                    StreamData(
                        originator=self.name,
                        force_end_call=True,
                    )
                )

    @override
    async def receive_task(self):
        try:
            while not self.stop_event.is_set():
                t = asyncio.create_task(self.session.recv())
                await self.wait_respecting_shutdown(t)
                if not t.done():
                    continue
                raw_msg = t.result()
                msg = json.loads(raw_msg)
                await self._handle_message(msg)
        except websockets.exceptions.ConnectionClosedOK:
            if not self.stop_event.is_set():
                await self.receive_queue.put(
                    StreamData(
                        originator=self.name,
                        force_end_call=True,
                    )
                )

    async def _handle_message(self, message: dict):
        """Handle incoming WebSocket messages."""
        msg_type = message.get("type")

        if msg_type == "conversation_initiation_metadata":
            event = message["conversation_initiation_metadata_event"]
            assert self._conversation_id is None
            self._conversation_id = event["conversation_id"]

        elif msg_type == "audio":
            event = message["audio_event"]
            if int(event["event_id"]) <= self._last_interrupt_id:
                return
            audio = base64.b64decode(event["audio_base_64"])
            stream_data = StreamData(
                originator=self.name,
                blob=Blob(data=audio, mime_type="audio/pcm"),
            )
            await self.receive_queue.put(stream_data)

        elif msg_type == "agent_response":
            event = message["agent_response_event"]
            stream_data = StreamData(
                originator=self.name,
                blob=None,
                output_transcription=event["agent_response"].strip(),
            )
            await self.receive_queue.put(stream_data)

        elif msg_type == "agent_response_correction":
            event = message["agent_response_correction_event"]
            stream_data = StreamData(
                originator=self.name,
                output_transcription_correction=TranscriptCorrection(
                    original=event["original_agent_response"].strip(),
                    corrected=event["corrected_agent_response"].strip(),
                ),
            )
            await self.receive_queue.put(stream_data)

        elif msg_type == "user_transcript":
            event = message["user_transcription_event"]
            stream_data = StreamData(
                originator=self.name,
                input_transcription=event["user_transcript"].strip(),
            )
            await self.receive_queue.put(stream_data)

        elif msg_type == "interruption":
            event = message["interruption_event"]
            self._last_interrupt_id = int(event["event_id"])
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.receive_queue.empty() and not self.call_ended:
                self.receive_queue.get_nowait()

        elif msg_type == "ping":
            event = message["ping_event"]
            await self.session.send(
                json.dumps(
                    {
                        "type": "pong",
                        "event_id": event["event_id"],
                    }
                )
            )

        elif msg_type == "client_tool_call":
            tool_call = message.get("client_tool_call", {})
            tool_name = tool_call.get("tool_name")
            if tool_name == "task_complete":
                await self.receive_queue.put(
                    StreamData(
                        originator=self.name,
                        force_end_call=True,
                    )
                )
            else:
                parameters = {
                    "tool_call_id": tool_call["tool_call_id"],
                    **tool_call.get("parameters", {}),
                }

                logging.error(
                    f"Unknown Tool call: {tool_name} with parameters: {parameters}"
                )
        else:
            logging.error(f"Unknown message type: {msg_type}")

    @override
    async def close(self):
        logging.info("Closing elevenlabs conversation")
        await self.session.close()
        self.session = None
