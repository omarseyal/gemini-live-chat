#!/usr/bin/env python3
"""
Reachy Mini Quiz Game for 3rd Graders
Reachy asks questions, you answer, and Reachy reacts!
"""

import os
import time
import requests
from dotenv import load_dotenv
from google import genai

load_dotenv()

REACHY_URL = "http://reachy-mini.local:8000"


class ReachyQuizGame:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.score = 0
        self.questions_asked = 0

    def reachy_move(self, antennas=(0, 0), head_pitch=0, duration=0.5):
        """Move Reachy's antennas and head."""
        try:
            requests.post(f"{REACHY_URL}/api/move/goto", json={
                "antennas": list(antennas),
                "head_pose": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": head_pitch, "yaw": 0},
                "duration": duration,
                "interpolation": "minjerk"
            }, timeout=5)
            time.sleep(duration + 0.1)
        except Exception as e:
            pass  # Continue even if movement fails

    def reachy_thinking(self):
        """Reachy thinking animation."""
        self.reachy_move(antennas=(0.3, -0.3), head_pitch=0.2, duration=0.3)

    def reachy_excited(self):
        """Reachy excited/correct animation."""
        for _ in range(2):
            self.reachy_move(antennas=(0.6, 0.6), duration=0.2)
            self.reachy_move(antennas=(-0.2, -0.2), duration=0.2)
        self.reachy_move(antennas=(0, 0), duration=0.3)

    def reachy_sad(self):
        """Reachy sad/incorrect animation."""
        self.reachy_move(antennas=(-0.4, -0.4), head_pitch=-0.15, duration=0.5)
        time.sleep(0.5)
        self.reachy_move(antennas=(0, 0), head_pitch=0, duration=0.5)

    def reachy_neutral(self):
        """Reset to neutral position."""
        self.reachy_move(antennas=(0, 0), head_pitch=0, duration=0.3)

    def generate_question(self):
        """Generate a 3rd grade appropriate question."""
        self.reachy_thinking()

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents="""Generate a fun trivia question appropriate for 3rd graders (ages 8-9).
Topics can include: basic math, animals, science, geography, history, or fun facts.
Keep it simple and engaging.

Respond in this exact format:
QUESTION: [your question here]
ANSWER: [the correct answer - keep it short, 1-3 words]
HINT: [a helpful hint]

Example:
QUESTION: How many legs does a spider have?
ANSWER: 8
HINT: It's more than 6 but less than 10!"""
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
        """Use Gemini to check if the answer is correct (handles variations)."""
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
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

        user_answer = input("\n✏️  Your answer: ").strip()

        if not user_answer:
            print("No answer provided!")
            return True

        print("\n🤔 Reachy is checking your answer...")
        self.reachy_thinking()

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
        again = input("\n🎮 Play another round? (yes/no): ").strip().lower()
        return again in ['yes', 'y', 'yeah', 'sure', 'ok']

    def run(self):
        """Run the quiz game."""
        print("\n" + "=" * 50)
        print("🎓 Welcome to Reachy's Quiz Game! 🎓")
        print("=" * 50)
        print("\nHi! I'm Reachy, your robot quiz master!")
        print("I'll ask you questions perfect for 3rd graders.")
        print("Let's see how many you can get right!\n")

        self.reachy_excited()

        try:
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
            self.reachy_neutral()


def main():
    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not set in .env file")
        return

    # Check Reachy connection
    try:
        resp = requests.get(f"{REACHY_URL}/api/daemon/status", timeout=5)
        if resp.json().get('state') != 'running':
            print("Warning: Reachy Mini daemon not running. Animations may not work.")
    except:
        print("Warning: Could not connect to Reachy Mini. Animations disabled.")

    game = ReachyQuizGame()
    game.run()


if __name__ == "__main__":
    main()
