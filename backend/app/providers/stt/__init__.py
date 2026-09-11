"""STT provider interfaces and adapters."""

from app.providers.stt.deepgram import DeepgramAdapter
from app.providers.stt.interface import STTInterface

__all__ = ["DeepgramAdapter", "STTInterface"]
