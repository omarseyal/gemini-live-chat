#!/usr/bin/env python3
"""
Simple Reachy Quiz - Voice Input
Speak your answer to Reachy!
"""

import os
import sys
import io
import time

print("Step 1: Starting...", flush=True)

# Load API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY="):
                    GEMINI_API_KEY = line.strip().split("=", 1)[1]
                    break

if not GEMINI_API_KEY:
    print("ERROR: No GEMINI_API_KEY found!")
    sys.exit(1)

print("Step 2: API key loaded", flush=True)

# Imports
from google import genai
from google.genai import types
print("Step 3: Google genai imported", flush=True)

import numpy as np
from scipy.io import wavfile
print("Step 4: Numpy/scipy imported", flush=True)

from reachy_mini import ReachyMini
print("Step 5: Reachy mini imported", flush=True)

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)
print("Step 6: Gemini client created", flush=True)

# Connect to Reachy
print("Step 7: Connecting to Reachy...", flush=True)
reachy = ReachyMini(media_backend="gstreamer")
reachy.__enter__()
print("Step 8: Reachy connected!", flush=True)

# Start audio
reachy.media.start_recording()
print("Step 9: Audio recording started", flush=True)


def record_audio(duration_seconds=4):
    """Record audio from Reachy's microphone."""
    print(f"\n🎤 Listening for {duration_seconds} seconds... Speak now!", flush=True)

    sample_rate = reachy.media.get_input_audio_samplerate()
    all_samples = []
    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        samples = reachy.media.get_audio_sample()
        if samples is not None and len(samples) > 0:
            all_samples.append(samples)
        time.sleep(0.05)

    if not all_samples:
        return None, sample_rate

    # Concatenate and convert to mono
    audio_data = np.concatenate(all_samples, axis=0)
    if len(audio_data.shape) > 1 and audio_data.shape[1] == 2:
        audio_data = np.mean(audio_data, axis=1)

    print(f"   Recorded {len(audio_data)} samples", flush=True)
    return audio_data.astype(np.float32), sample_rate


def audio_to_wav_bytes(audio_data, sample_rate):
    """Convert audio to WAV bytes."""
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, int(sample_rate), audio_int16)
    wav_buffer.seek(0)
    return wav_buffer.read()


def transcribe_audio(audio_data, sample_rate):
    """Use Gemini to transcribe audio."""
    wav_bytes = audio_to_wav_bytes(audio_data, sample_rate)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="audio/wav",
                            data=wav_bytes
                        )
                    ),
                    types.Part(text="Transcribe this audio. Return ONLY the spoken words. If unclear or no speech, return 'UNCLEAR'.")
                ]
            )
        ]
    )
    return response.text.strip()


# Main quiz
print("\n" + "="*50, flush=True)
print("🎓 Reachy's Voice Quiz for 3rd Graders! 🎓", flush=True)
print("="*50, flush=True)

try:
    # Generate question
    print("\n🤔 Thinking of a question...", flush=True)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Generate one simple trivia question for a 3rd grader. Just the question, nothing else."
    )
    question = response.text.strip()
    print(f"\n📚 Question: {question}", flush=True)

    # Record voice answer
    audio_data, sample_rate = record_audio(duration_seconds=5)

    if audio_data is None:
        print("❌ No audio recorded!", flush=True)
    else:
        # Transcribe
        print("\n🔄 Processing your answer...", flush=True)
        user_answer = transcribe_audio(audio_data, sample_rate)
        print(f"   I heard: \"{user_answer}\"", flush=True)

        if user_answer and user_answer != "UNCLEAR":
            # Check answer
            check_response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"Question: {question}\nUser answer: {user_answer}\n\nIs this correct? Reply CORRECT or INCORRECT with a brief kid-friendly explanation."
            )
            print(f"\n🎯 {check_response.text.strip()}", flush=True)
        else:
            print("❌ Couldn't understand. Try again!", flush=True)

except KeyboardInterrupt:
    print("\n\nQuiz ended.", flush=True)
finally:
    # Cleanup
    print("\nCleaning up...", flush=True)
    reachy.media.stop_recording()
    reachy.__exit__(None, None, None)
    print("Done! 👋", flush=True)
