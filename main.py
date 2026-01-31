#!/usr/bin/env python3
"""
Reachy Mini + Gemini Live Chat

Streams audio and video from Reachy Mini to Google Gemini Live API
and plays back Gemini's audio responses through the robot's speaker.
"""

import asyncio
import os
import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from reachy_mini import ReachyMini
from scipy.signal import resample

load_dotenv()

# Configuration
GEMINI_MODEL = "gemini-2.5-flash-preview-native-audio-dialog"
SAMPLE_RATE_INPUT = 16000  # Gemini expects 16kHz mono
SAMPLE_RATE_OUTPUT = 24000  # Gemini outputs 24kHz
VIDEO_FPS = 1  # Gemini processes at 1 FPS
VIDEO_SIZE = (768, 768)  # Optimal resolution for Gemini

SYSTEM_INSTRUCTION = """You are a friendly robot assistant named Reachy.
You can see through your camera and hear through your microphone.
You speak naturally and conversationally. Keep responses concise.
You're helpful, curious, and have a warm personality."""


class ReachyGeminiChat:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.reachy = None
        self.session = None
        self.running = False
        self.audio_output_queue = asyncio.Queue()

    async def start(self):
        """Initialize Reachy and connect to Gemini."""
        print("Connecting to Reachy Mini...")
        self.reachy = ReachyMini(media_backend="default")
        self.reachy.__enter__()

        # Start audio devices
        self.reachy.media.start_recording()
        self.reachy.media.start_playing()

        print("Connected to Reachy Mini")
        print(f"Audio input sample rate: {self.reachy.media.get_input_audio_samplerate()} Hz")
        print(f"Audio output sample rate: {self.reachy.media.get_output_audio_samplerate()} Hz")

        # Connect to Gemini Live API
        print("Connecting to Gemini Live API...")
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": SYSTEM_INSTRUCTION,
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": "Kore"  # Options: Puck, Charon, Kore, Fenrir, Aoede
                    }
                }
            }
        }

        self.session = await self.client.aio.live.connect(
            model=GEMINI_MODEL,
            config=config
        )
        print("Connected to Gemini Live API")
        self.running = True

    async def stop(self):
        """Clean up resources."""
        self.running = False

        if self.session:
            await self.session.close()

        if self.reachy:
            self.reachy.media.stop_recording()
            self.reachy.media.stop_playing()
            self.reachy.__exit__(None, None, None)

        print("Disconnected")

    async def capture_and_send_audio(self):
        """Capture audio from Reachy and send to Gemini."""
        reachy_sample_rate = self.reachy.media.get_input_audio_samplerate()

        while self.running:
            try:
                # Get audio samples from Reachy (shape: samples x 2, float32, 16kHz)
                samples = self.reachy.media.get_audio_sample()

                if samples is None or len(samples) == 0:
                    await asyncio.sleep(0.01)
                    continue

                # Convert stereo to mono by averaging channels
                if len(samples.shape) > 1 and samples.shape[1] == 2:
                    mono_samples = np.mean(samples, axis=1)
                else:
                    mono_samples = samples.flatten()

                # Resample if needed (Reachy is 16kHz, Gemini expects 16kHz)
                if reachy_sample_rate != SAMPLE_RATE_INPUT:
                    num_samples = int(len(mono_samples) * SAMPLE_RATE_INPUT / reachy_sample_rate)
                    mono_samples = resample(mono_samples, num_samples)

                # Convert to 16-bit PCM
                pcm_data = (mono_samples * 32767).astype(np.int16).tobytes()

                # Send to Gemini
                await self.session.send_realtime_input(
                    audio=types.Blob(data=pcm_data, mime_type="audio/pcm;rate=16000")
                )

            except Exception as e:
                print(f"Audio capture error: {e}")
                await asyncio.sleep(0.1)

    async def capture_and_send_video(self):
        """Capture video frames from Reachy and send to Gemini at 1 FPS."""
        while self.running:
            try:
                # Get frame from Reachy camera
                frame = self.reachy.media.get_frame()

                if frame is None:
                    await asyncio.sleep(0.1)
                    continue

                # Resize to optimal resolution
                frame_resized = cv2.resize(frame, VIDEO_SIZE)

                # Encode as JPEG
                _, jpeg_buffer = cv2.imencode('.jpg', frame_resized,
                                               [cv2.IMWRITE_JPEG_QUALITY, 80])

                # Send to Gemini
                await self.session.send_realtime_input(
                    media=types.Blob(data=jpeg_buffer.tobytes(), mime_type="image/jpeg")
                )

                # Wait for next frame (1 FPS)
                await asyncio.sleep(1.0 / VIDEO_FPS)

            except Exception as e:
                print(f"Video capture error: {e}")
                await asyncio.sleep(0.5)

    async def receive_and_play_audio(self):
        """Receive audio from Gemini and play through Reachy speaker."""
        reachy_output_rate = self.reachy.media.get_output_audio_samplerate()

        while self.running:
            try:
                turn = self.session.receive()

                async for response in turn:
                    if not self.running:
                        break

                    # Check for audio data
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                audio_data = part.inline_data.data

                                # Convert from bytes to float32
                                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                                audio_float = audio_array.astype(np.float32) / 32767.0

                                # Resample from 24kHz to Reachy's output rate
                                if reachy_output_rate != SAMPLE_RATE_OUTPUT:
                                    num_samples = int(len(audio_float) * reachy_output_rate / SAMPLE_RATE_OUTPUT)
                                    audio_float = resample(audio_float, num_samples)

                                # Convert to stereo if needed
                                audio_stereo = np.column_stack([audio_float, audio_float])

                                # Play through Reachy
                                self.reachy.media.push_audio_sample(audio_stereo)

                    # Check for turn completion
                    if response.server_content and response.server_content.turn_complete:
                        print("[Turn complete]")

            except Exception as e:
                if self.running:
                    print(f"Audio receive error: {e}")
                await asyncio.sleep(0.1)

    async def run(self):
        """Main loop - run all tasks concurrently."""
        try:
            await self.start()

            print("\n" + "="*50)
            print("Reachy Mini + Gemini Live Chat")
            print("="*50)
            print("Speak to Reachy - Gemini can see and hear you!")
            print("Press Ctrl+C to exit")
            print("="*50 + "\n")

            # Run all tasks concurrently
            await asyncio.gather(
                self.capture_and_send_audio(),
                self.capture_and_send_video(),
                self.receive_and_play_audio(),
            )

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            await self.stop()


async def main():
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Create a .env file with: GEMINI_API_KEY=your_api_key_here")
        return

    chat = ReachyGeminiChat()
    await chat.run()


if __name__ == "__main__":
    asyncio.run(main())
