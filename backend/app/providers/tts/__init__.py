"""TTS provider interfaces and adapters."""

from app.providers.tts.azure import AzureTTSAdapter
from app.providers.tts.elevenlabs import ElevenLabsAdapter
from app.providers.tts.interface import TTSInterface

__all__ = ["AzureTTSAdapter", "ElevenLabsAdapter", "TTSInterface"]
