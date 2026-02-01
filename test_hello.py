#!/usr/bin/env python3
"""
Simple test: Reachy Mini + Gemini Flash API
"""

import os
import requests
from dotenv import load_dotenv
from google import genai

load_dotenv()

REACHY_URL = "http://reachy-mini.local:8000"

def test_reachy_connection():
    """Test connection to Reachy Mini."""
    print("Testing Reachy Mini connection...")
    try:
        resp = requests.get(f"{REACHY_URL}/api/daemon/status", timeout=5)
        status = resp.json()
        print(f"  Robot: {status.get('robot_name', 'Unknown')}")
        print(f"  State: {status.get('state', 'Unknown')}")
        print(f"  Wireless: {status.get('wireless_version', False)}")
        return status.get('state') == 'running'
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_gemini():
    """Test Gemini Flash API."""
    print("\nTesting Gemini Flash API...")
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say hello in one short sentence as a friendly robot named Reachy."
        )
        message = response.text
        print(f"  Gemini says: {message}")
        return message
    except Exception as e:
        print(f"  Error: {e}")
        return None

def make_reachy_wave():
    """Make Reachy do a wave animation with antennas."""
    print("\nMaking Reachy wave...")
    try:
        # Move antennas up
        requests.post(f"{REACHY_URL}/api/move/goto", json={
            "antennas": [0.5, 0.5],  # radians
            "duration": 0.5,
            "interpolation": "minjerk"
        }, timeout=5)

        import time
        time.sleep(0.6)

        # Move antennas down
        requests.post(f"{REACHY_URL}/api/move/goto", json={
            "antennas": [-0.3, -0.3],
            "duration": 0.5,
            "interpolation": "minjerk"
        }, timeout=5)

        time.sleep(0.6)

        # Return to neutral
        requests.post(f"{REACHY_URL}/api/move/goto", json={
            "antennas": [0.0, 0.0],
            "duration": 0.5,
            "interpolation": "minjerk"
        }, timeout=5)

        print("  Wave complete!")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    print("=" * 50)
    print("Reachy Mini + Gemini Flash Test")
    print("=" * 50)

    # Test Reachy connection
    reachy_ok = test_reachy_connection()

    # Test Gemini
    gemini_response = test_gemini()

    # If both work, make Reachy wave
    if reachy_ok and gemini_response:
        make_reachy_wave()
        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("Some tests failed - check output above")
        print("=" * 50)

if __name__ == "__main__":
    main()
