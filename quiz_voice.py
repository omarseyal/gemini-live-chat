#!/usr/bin/env python3
"""
Reachy Mini Voice Quiz Game for 3rd Graders
Speak your answers to Reachy!
"""

import os
import io
import time
import base64
import asyncio
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from reachy_mini import ReachyMini
from scipy.io import wavfile

load_dotenv()

GEMINI_MODEL = "gemini-2.0-flash"


class ReachyVoiceQuiz:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.reachy = None
        self.score = 0
        self.questions_asked = 0

    def start_reachy(self):
        """Initialize Reachy Mini."""
        print("Connecting to Reachy Mini...")
        self.reachy = ReachyMini(media_backend="gstreamer")
        self.reachy.__enter__()
        self.reachy.media.start_recording()
        self.reachy.media.start_playing()
        print("Reachy Mini connected!")

    def stop_reachy(self):
        """Cleanup Reachy."""
        if self.reachy:
            self.reachy.media.stop_recording()
            self.reachy.media.stop_playing()
            self.reachy.__exit__(None, None, None)

    def reachy_move(self, antennas=(0, 0), head_pitch=0, duration=0.5):
        """Move Reachy's antennas and head."""
        try:
            self.reachy.goto_target(
                antennas=np.array(antennas),
                head=np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, head_pitch * 0.01],  # z offset for pitch effect
                              [0, 0, 0, 1]]),
                duration=duration,
                method="minjerk"
            )
            time.sleep(duration + 0.1)
        except Exception as e:
            print(f"Movement error: {e}")

    def reachy_thinking(self):
        """Reachy thinking animation."""
        self.reachy_move(antennas=(0.3, -0.3), duration=0.3)

    def reachy_excited(self):
        """Reachy excited/correct animation."""
        for _ in range(2):
            self.reachy_move(antennas=(0.6, 0.6), duration=0.2)
            self.reachy_move(antennas=(-0.2, -0.2), duration=0.2)
        self.reachy_move(antennas=(0, 0), duration=0.3)

    def reachy_sad(self):
        """Reachy sad/incorrect animation."""
        self.reachy_move(antennas=(-0.4, -0.4), head_pitch=-15, duration=0.5)
        time.sleep(0.5)
        self.reachy_move(antennas=(0, 0), head_pitch=0, duration=0.5)

    def reachy_listening(self):
        """Reachy listening animation."""
        self.reachy_move(antennas=(0.2, 0.2), head_pitch=5, duration=0.3)

    def reachy_neutral(self):
        """Reset to neutral position."""
        self.reachy_move(antennas=(0, 0), head_pitch=0, duration=0.3)

    def record_audio(self, duration_seconds=5):
        """Record audio from Reachy's microphone."""
        print(f"🎤 Listening for {duration_seconds} seconds... Speak now!")
        self.reachy_listening()

        # Collect audio samples
        sample_rate = self.reachy.media.get_input_audio_samplerate()
        all_samples = []
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            samples = self.reachy.media.get_audio_sample()
            if samples is not None and len(samples) > 0:
                all_samples.append(samples)
            time.sleep(0.05)

        if not all_samples:
            return None, sample_rate

        # Concatenate all samples
        audio_data = np.concatenate(all_samples, axis=0)

        # Convert to mono if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[1] == 2:
            audio_data = np.mean(audio_data, axis=1)

        self.reachy_neutral()
        return audio_data.astype(np.float32), sample_rate

    def audio_to_wav_bytes(self, audio_data, sample_rate):
        """Convert audio data to WAV bytes."""
        # Normalize and convert to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Create WAV in memory
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, sample_rate, audio_int16)
        wav_buffer.seek(0)
        return wav_buffer.read()

    def transcribe_audio(self, audio_data, sample_rate):
        """Use Gemini to transcribe audio."""
        wav_bytes = self.audio_to_wav_bytes(audio_data, sample_rate)

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="audio/wav",
                                data=wav_bytes
                            )
                        ),
                        types.Part(text="Transcribe this audio. Return ONLY the spoken words, nothing else. If you can't understand it or there's no speech, return 'UNCLEAR'.")
                    ]
                )
            ]
        )

        return response.text.strip()

    def generate_question(self):
        """Generate a 3rd grade appropriate question."""
        self.reachy_thinking()

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents="""Generate a fun trivia question appropriate for 3rd graders (ages 8-9).
Topics can include: basic math, animals, science, geography, history, or fun facts.
Keep it simple and engaging.

Respond in this exact format:
QUESTION: [your question here]
ANSWER: [the correct answer - keep it short, 1-3 words]
HINT: [a helpful hint]"""
        )

        text = response.text
        lines = text.strip().split('\n')

        question = ""
        answer = ""
        hint = ""

        for line in lines:
            if line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()
            elif line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()
            elif line.startswith("HINT:"):
                hint = line.replace("HINT:", "").strip()

        self.reachy_neutral()
        return question, answer, hint

    def check_answer(self, user_answer, correct_answer, question):
        """Use Gemini to check if the answer is correct."""
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"""Question: {question}
Correct answer: {correct_answer}
User's answer: {user_answer}

Is the user's answer correct? Consider spelling variations, synonyms, and equivalent answers.
Respond with only "CORRECT" or "INCORRECT" followed by a brief encouraging message for a 3rd grader.

Example responses:
CORRECT Great job! You really know your stuff!
INCORRECT Not quite, but good try! The answer is 8."""
        )

        result = response.text.strip()
        is_correct = result.upper().startswith("CORRECT")
        message = result.split(" ", 1)[1] if " " in result else ""

        return is_correct, message

    def play_round(self):
        """Play one round of the quiz."""
        print("\n" + "=" * 50)
        print("🤖 Reachy is thinking of a question...")
        print("=" * 50)

        question, correct_answer, hint = self.generate_question()
        self.questions_asked += 1

        print(f"\n📚 Question #{self.questions_asked}:")
        print(f"   {question}")
        print(f"\n💡 Hint: {hint}")

        # Record voice answer
        print("\n🎤 Say your answer to Reachy!")
        audio_data, sample_rate = self.record_audio(duration_seconds=5)

        if audio_data is None:
            print("❌ No audio recorded!")
            return True

        print("\n🤔 Reachy is processing your answer...")
        self.reachy_thinking()

        # Transcribe audio
        user_answer = self.transcribe_audio(audio_data, sample_rate)
        print(f"   I heard: \"{user_answer}\"")

        if user_answer == "UNCLEAR" or not user_answer:
            print("❌ Couldn't understand that. Let's try another question!")
            self.reachy_neutral()
            return True

        # Check answer
        is_correct, message = self.check_answer(user_answer, correct_answer, question)

        if is_correct:
            self.score += 1
            print(f"\n✅ CORRECT! {message}")
            self.reachy_excited()
        else:
            print(f"\n❌ {message}")
            print(f"   The correct answer was: {correct_answer}")
            self.reachy_sad()

        print(f"\n📊 Score: {self.score}/{self.questions_asked}")

        # Ask if they want to continue
        print("\n🎮 Say 'yes' for another question, or 'no' to stop.")
        audio_data, sample_rate = self.record_audio(duration_seconds=3)

        if audio_data is not None:
            response = self.transcribe_audio(audio_data, sample_rate).lower()
            return "yes" in response or "yeah" in response or "sure" in response or "ok" in response

        return False

    def run(self):
        """Run the quiz game."""
        print("\n" + "=" * 50)
        print("🎓 Welcome to Reachy's Voice Quiz Game! 🎓")
        print("=" * 50)
        print("\nHi! I'm Reachy, your robot quiz master!")
        print("I'll ask you questions perfect for 3rd graders.")
        print("Speak your answers - I'm listening!\n")

        try:
            self.start_reachy()
            self.reachy_excited()

            playing = True
            while playing:
                playing = self.play_round()

            # Game over
            print("\n" + "=" * 50)
            print("🏆 GAME OVER! 🏆")
            print("=" * 50)
            print(f"\nFinal Score: {self.score}/{self.questions_asked}")

            if self.questions_asked > 0:
                percentage = (self.score / self.questions_asked) * 100
                if percentage >= 80:
                    print("🌟 Amazing! You're a quiz champion!")
                    self.reachy_excited()
                elif percentage >= 60:
                    print("👍 Great job! Keep learning!")
                    self.reachy_excited()
                else:
                    print("💪 Good effort! Practice makes perfect!")
                    self.reachy_neutral()

            print("\nThanks for playing with Reachy! Goodbye! 👋\n")

        except KeyboardInterrupt:
            print("\n\nGame ended early. Thanks for playing!")
        finally:
            self.reachy_neutral()
            self.stop_reachy()


def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not set")
        print("Create a .env file with: GEMINI_API_KEY=your_key_here")
        return

    game = ReachyVoiceQuiz()
    game.run()


if __name__ == "__main__":
    main()
