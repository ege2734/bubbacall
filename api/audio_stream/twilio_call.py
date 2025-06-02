import asyncio
import base64
import json
import logging
from typing import override

import websockets
from fastapi import WebSocket
from google.genai.types import Blob

from api.audio_stream.stream_data import StreamData
from api.audio_stream.stream_operator import StreamOperator


class TwilioCall(StreamOperator):
    """
    `Send` sends audio to Twilio.
    `Receive` receives audio from Twilio.

    This class does not own `ws`, and therefore WILL NOT close it when it is done.
    More info at: https://www.twilio.com/docs/voice/media-streams/websocket-messages#send-websocket-messages-to-twilio

    """

    def __init__(
        self,
        ws: WebSocket,
    ):
        super().__init__(
            "twilio_call",
        )
        self.ws = ws
        self.stream_sid = None

    @override
    async def initialize(self):
        async for raw_msg in self.ws.iter_text():
            message = json.loads(raw_msg)
            if message["event"] == "connected":
                logging.error("Confirmed connection")
            elif message["event"] == "start":
                self.stream_sid = message["start"]["streamSid"]
                break
            else:
                logging.error(f"Received unexpected message: {message}")
        assert self.stream_sid is not None

    @override
    async def send_task(self):
        try:
            while not self.stop_event.is_set():
                stream_data = await self.get_from_send_queue()
                if stream_data is None or stream_data.blob is None:
                    continue
                await self.ws.send_json(
                    {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {
                            "payload": base64.b64encode(stream_data.blob.data).decode()
                        },
                    }
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
                t = asyncio.create_task(self.ws.receive_text())
                await self.wait_respecting_shutdown(t)
                if not t.done():
                    continue
                raw_msg = t.result()
                msg = json.loads(raw_msg)
                if msg["event"] == "media":
                    await self.receive_queue.put(
                        StreamData(
                            originator=self.name,
                            blob=Blob(data=base64.b64decode(msg["media"]["payload"])),
                        )
                    )
                    continue
                logging.error(f"Received unexpected message: {msg}")
        except websockets.exceptions.ConnectionClosedOK:
            if not self.stop_event.is_set():
                await self.receive_queue.put(
                    StreamData(
                        originator=self.name,
                        force_end_call=True,
                    )
                )

    @override
    async def close(self):
        logging.info("Closing twilio connection")
