import asyncio
import base64
import json
import logging
import os
import queue
import threading
import wave
from typing import Any, Dict, Optional

import pyaudio
import websockets

from api.utils.settings import get_setting

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElevenLabsVoiceAgent:
    def __init__(self, agent_id: str, api_key: str):
        """
        Initialize the ElevenLabs Voice Agent

        Args:
            agent_id: The agent ID for your ElevenLabs conversational AI agent
            api_key: Your ElevenLabs API key (optional for public agents)
        """
        self.agent_id = agent_id
        self.api_key = api_key

        # Audio configuration
        self.sample_rate = 16000  # 16kHz for speech
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        self.channels = 1

        # PyAudio instance
        self.audio = pyaudio.PyAudio()

        # Audio queues
        self.audio_input_queue = queue.Queue()
        self.audio_output_queue = queue.Queue()

        # Control flags
        self.is_recording = False
        self.is_playing = False
        self.websocket = None

        # Threads
        self.input_thread = None
        self.output_thread = None

    def get_websocket_url(self) -> str:
        """Get the WebSocket URL for connecting to ElevenLabs"""
        base_url = "wss://api.elevenlabs.io/v1/convai/conversation"
        return f"{base_url}?agent_id={self.agent_id}"

    def get_headers(self) -> Dict[str, str]:
        """Get headers for WebSocket connection"""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def connect(self):
        """Establish WebSocket connection to ElevenLabs"""
        url = self.get_websocket_url()
        headers = self.get_headers()

        try:
            self.websocket = await websockets.connect(
                url, extra_headers=headers, ping_interval=30, ping_timeout=10
            )
            logger.info("Connected to ElevenLabs WebSocket")

            # Send initial conversation data
            await self.send_conversation_initiation()

        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise

    async def send_conversation_initiation(self):
        """Send conversation initiation message"""
        init_message = {
            "type": "conversation_initiation_client_data",
            "conversation_config_override": {"agent": {"language": "en"}},
        }
        await self.websocket.send(json.dumps(init_message))
        logger.info("Sent conversation initiation")

    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk to ElevenLabs"""
        # Convert audio data to base64
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        message = {"type": "user_audio_chunk", "chunk": audio_b64}

        if self.websocket:
            await self.websocket.send(json.dumps(message))

    async def handle_incoming_message(self, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "audio_response":
                # Handle audio response from agent
                audio_data = base64.b64decode(data["chunk"])
                self.audio_output_queue.put(audio_data)

            elif message_type == "user_transcript":
                # Handle user transcript
                transcript = data.get("transcript", "")
                logger.info(f"User said: {transcript}")

            elif message_type == "agent_response":
                # Handle agent text response
                response = data.get("response", "")
                logger.info(f"Agent response: {response}")

            elif message_type == "conversation_initiation_metadata":
                # Handle conversation metadata
                logger.info("Conversation initiated successfully")

            elif message_type == "ping":
                # Respond to ping with pong
                pong_message = {"type": "pong"}
                await self.websocket.send(json.dumps(pong_message))

            else:
                logger.debug(f"Received message type: {message_type}")

        except json.JSONDecodeError:
            logger.error("Failed to decode JSON message")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def start_audio_input(self):
        """Start recording audio from microphone"""

        def record_audio():
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )

            logger.info("Started recording audio")

            try:
                while self.is_recording:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_input_queue.put(data)
            except Exception as e:
                logger.error(f"Error recording audio: {e}")
            finally:
                stream.stop_stream()
                stream.close()
                logger.info("Stopped recording audio")

        self.is_recording = True
        self.input_thread = threading.Thread(target=record_audio)
        self.input_thread.start()

    def start_audio_output(self):
        """Start playing audio from speaker"""

        def play_audio():
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
            )

            logger.info("Started audio playback")

            try:
                while self.is_playing:
                    try:
                        audio_data = self.audio_output_queue.get(timeout=0.1)
                        stream.write(audio_data)
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error playing audio: {e}")
            finally:
                stream.stop_stream()
                stream.close()
                logger.info("Stopped audio playback")

        self.is_playing = True
        self.output_thread = threading.Thread(target=play_audio)
        self.output_thread.start()

    async def process_audio_input(self):
        """Process audio input queue and send to WebSocket"""
        while self.is_recording:
            try:
                audio_data = self.audio_input_queue.get(timeout=0.1)
                await self.send_audio_chunk(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio input: {e}")

    async def listen_for_messages(self):
        """Listen for incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                await self.handle_incoming_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error listening for messages: {e}")

    async def run(self):
        """Main run loop for the voice agent"""
        try:
            # Connect to WebSocket
            await self.connect()

            # Start audio input/output threads
            self.start_audio_input()
            self.start_audio_output()

            # Create tasks for audio processing and message listening
            audio_task = asyncio.create_task(self.process_audio_input())
            listen_task = asyncio.create_task(self.listen_for_messages())

            # Wait for tasks to complete
            await asyncio.gather(audio_task, listen_task)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")

        # Stop recording and playing
        self.is_recording = False
        self.is_playing = False

        # Wait for threads to finish
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=2)

        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=2)

        # Close WebSocket connection
        if self.websocket:
            await self.websocket.close()

        # Terminate PyAudio
        self.audio.terminate()

        logger.info("Cleanup complete")


async def main():
    """Main function to run the voice agent"""
    # Replace with your actual agent ID
    AGENT_ID = "agent_01jwcdkfj3fmdr2ypdmrtnmzgt"

    agent = ElevenLabsVoiceAgent(
        agent_id=AGENT_ID, api_key=get_setting("ELEVENLABS_API_KEY")
    )

    print("Starting ElevenLabs Voice Agent...")
    print("Press Ctrl+C to stop")

    try:
        await agent.run()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    asyncio.run(main())
