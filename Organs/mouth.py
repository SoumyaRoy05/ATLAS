import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from kokoro_onnx import Kokoro

import sounddevice as sd


# -----------------------------------------------------------------------------
# 1. Environment & Model Paths
# -----------------------------------------------------------------------------
load_dotenv()

# Enforce Kokoro model and voices paths from environment variables or defaults
KOKORO_DIR = Path(os.getenv("KOKORO_DIR", r"D:\Codes\AI_MODELS\kokoro"))
DEFAULT_MODEL_PATH = Path(os.getenv("KOKORO_MODEL_PATH", KOKORO_DIR / "kokoro-v0_19.onnx"))
DEFAULT_VOICES_PATH = Path(os.getenv("KOKORO_VOICES_PATH", KOKORO_DIR / "voices.bin"))


# -----------------------------------------------------------------------------
# 2. Vocal Motor Organ Class
# -----------------------------------------------------------------------------
class Mouth:
    """Vocal output organ for Atlas.

    Attempts Kokoro TTS synthesis; gracefully falls back to console output
    if Kokoro is uninstalled, unconfigured, or encounters an error.
    """

    # uses the Kokoro TTS engine to synthesize speech from text
    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        voices_path: Optional[str | Path] = None,
        default_voice: str = "am_adam",
    ):
        self._is_speaking = False
        self.default_voice = default_voice
        self.kokoro = None

        resolved_model = str(model_path if model_path is not None else DEFAULT_MODEL_PATH)
        resolved_voices = str(voices_path if voices_path is not None else DEFAULT_VOICES_PATH)

        # Attempt Kokoro engine initialization
        try:

            if os.path.exists(resolved_model) and os.path.exists(resolved_voices):
                self.kokoro = Kokoro(resolved_model, resolved_voices)
                print(f"[Mouth]: Kokoro TTS loaded successfully from {resolved_model}")
            else:
                print("[Mouth]: Kokoro model files not found on disk. Falling back to console output.")
                self.kokoro = None

        except Exception as e:
            print(f"[Mouth]: Kokoro initialization failed ({e}). Falling back to console output.")
            self.kokoro = None


    # property is a read-only attribute that indicates whether the Mouth is currently speaking
    @property
    def is_speaking(self) -> bool:
        """Monitored by ears.py for wake-word barge-in interruption."""
        return self._is_speaking


    # stop method interrupts any ongoing speech playback immediately
    def stop(self) -> None:
        """Interrupts ongoing speech playback immediately."""
        self._is_speaking = False

        try:
            sd.stop()
        except Exception:
            pass


    # speak method synthesizes text using Kokoro if available; otherwise prints directly
    def speak(self, text: str) -> None:
        """Synthesizes text using Kokoro if available; otherwise prints directly."""
        if not text or not text.strip():
            return

        clean_text = " ".join(text.strip().split())

        # Attempt Kokoro TTS synthesis and playback
        try:
            if self.kokoro is None:
                raise RuntimeError("Kokoro engine is not initialized or weights are missing")

            self._is_speaking = True

            samples, sample_rate = self.kokoro.create(
                clean_text,
                voice=self.default_voice,
                speed=1.0,
                lang="en-us",
            )
            sd.play(samples, samplerate=sample_rate)
            sd.wait()

        
        except Exception as e:
            # Console fallback if synthesis or audio playback fails
            print(f"[Mouth]: Kokoro unavailable ({e}).")


        finally:
            self._is_speaking = False