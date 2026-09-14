#!/usr/bin/env python3
"""ElevenLabs API diagnostic script.

This script checks:
1. Subscription endpoint (GET /v1/user/subscription)
2. TTS endpoint (POST /v1/text-to-speech/{voice_id})

NEVER logs or prints the API key.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path so we can import config
sys.path.insert(0, str(Path(__file__).parent))

import httpx

from app.core.config import settings


_SUBSCRIPTION_URL = "https://api.elevenlabs.io/v1/user/subscription"
_TTS_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"  # API-compatible Free tier voice
_DEFAULT_MODEL = "eleven_flash_v2_5"
_TIMEOUT = 30.0


async def check_subscription(api_key: str) -> dict | None:
    """Check subscription endpoint."""
    print("\n=== Subscription Check ===")
    print(f"API key configured: {bool(api_key)}")

    if not api_key:
        print("ERROR: No API key configured. Set ELEVENLABS_API_KEY in .env")
        return None

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(_TIMEOUT),
        headers={"xi-api-key": api_key},
    ) as client:
        try:
            response = await client.get(_SUBSCRIPTION_URL)
            print(f"HTTP status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"tier: {data.get('tier', 'N/A')}")
                print(f"status: {data.get('status', 'N/A')}")
                print(f"character_count: {data.get('character_count', 'N/A')}")
                print(f"character_limit: {data.get('character_limit', 'N/A')}")
                print(f"max_character_limit_extension: {data.get('max_character_limit_extension', 'N/A')}")
                print(f"allowed_to_extend_character_limit: {data.get('allowed_to_extend_character_limit', 'N/A')}")
                print(f"can_extend_voice_limit: {data.get('can_extend_voice_limit', 'N/A')}")
                return data
            elif response.status_code == 401:
                print("ERROR: Authentication failed - API key is invalid")
                return None
            elif response.status_code == 403:
                print("ERROR: Authorization failed - API key lacks permissions")
                return None
            else:
                print(f"ERROR: Unexpected status {response.status_code}")
                try:
                    print(f"Response: {response.json()}")
                except Exception:
                    print(f"Response: {response.text[:200]}")
                return None

        except httpx.TimeoutException:
            print(f"ERROR: Request timed out after {_TIMEOUT}s")
            return None
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            return None


async def check_tts(api_key: str, voice_id: str, model: str) -> bool:
    """Check TTS endpoint with minimal text."""
    print("\n=== TTS Diagnostic Check ===")
    print(f"Voice ID: {voice_id}")
    print(f"Model: {model}")
    print(f"Text: 'Hello from Voice AI Lab.'")

    url = _TTS_URL_TEMPLATE.format(voice_id=voice_id)
    payload = {
        "text": "Hello from Voice AI Lab.",
        "model_id": model,
        "output_format": "mp3_44100_128",
    }

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(_TIMEOUT),
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    ) as client:
        try:
            response = await client.post(url, json=payload)
            print(f"HTTP status: {response.status_code}")

            if response.status_code == 200:
                content_type = response.headers.get("content-type", "unknown")
                audio_bytes = len(response.content)
                print(f"Content-Type: {content_type}")
                print(f"Audio bytes: {audio_bytes}")
                print("SUCCESS: TTS working!")
                return True
            else:
                # Extract error detail
                try:
                    body = response.json()
                    if isinstance(body, dict):
                        detail = body.get("detail", "")
                        if isinstance(detail, dict):
                            error_msg = detail.get("message", str(detail))
                        else:
                            error_msg = str(detail)
                    else:
                        error_msg = str(body)[:200]
                except Exception:
                    error_msg = response.text[:200] if response.text else "No error body"

                print(f"Provider error: {error_msg}")
                return False

        except httpx.TimeoutException:
            print(f"ERROR: Request timed out after {_TIMEOUT}s")
            return False
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            return False


async def main():
    """Run diagnostics."""
    print("=" * 50)
    print("ElevenLabs API Diagnostic")
    print("=" * 50)

    api_key = settings.elevenlabs_api_key
    voice_id = settings.default_tts_voice or _DEFAULT_VOICE
    model = settings.default_tts_model or _DEFAULT_MODEL

    print(f"\nConfiguration:")
    print(f"  API key configured: {bool(api_key)}")
    print(f"  Default voice: {voice_id}")
    print(f"  Default model: {model}")

    # Check subscription
    sub_data = await check_subscription(api_key)

    if sub_data is None:
        print("\n=== DIAGNOSIS FAILED ===")
        print("Could not retrieve subscription info.")
        return

    # Check TTS
    tts_ok = await check_tts(api_key, voice_id, model)

    # Final diagnosis
    print("\n" + "=" * 50)
    print("DIAGNOSIS SUMMARY")
    print("=" * 50)

    tier = sub_data.get("tier", "unknown")
    status = sub_data.get("status", "unknown")
    char_count = sub_data.get("character_count", 0)
    char_limit = sub_data.get("character_limit", 0)

    print(f"Subscription tier: {tier}")
    print(f"Subscription status: {status}")
    print(f"Character usage: {char_count}/{char_limit}")
    print(f"TTS endpoint: {'WORKING' if tts_ok else 'FAILED'}")

    if tier == "Free" and char_count < char_limit:
        print("\nAccount appears to have quota available.")
        if not tts_ok:
            print("But TTS is failing. Possible causes:")
            print("  - Model not available on free tier")
            print("  - Voice ID not available on free tier")
            print("  - API key has endpoint restrictions")
            print("  - Account requires email verification")
    elif char_count >= char_limit:
        print("\nCharacter quota exhausted!")
        print("Wait for reset or upgrade plan.")


if __name__ == "__main__":
    asyncio.run(main())
