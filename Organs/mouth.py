from dotenv import load_dotenv
from typing import Any, cast
import logging
import warnings

load_dotenv()

# Just removing the warnings to be printed in the console, as they are not relevant to the user and can be confusing.
class _WarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "unauthenticated requests to the HF Hub" not in record.getMessage()

logging.getLogger("huggingface_hub").addFilter(_WarningFilter())
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

warnings.filterwarnings(
    "ignore",
    message=r"You are sending unauthenticated requests to the HF Hub.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"dropout option adds dropout after all but last recurrent layer.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)

import sounddevice as sd
import numpy as np


# -----------------------------------------------------------------------------
# 1. Environment & Model Paths
# -----------------------------------------------------------------------------
load_dotenv()


# -----------------------------------------------------------------------------
# 2. Vocal Motor Organ Class
# -----------------------------------------------------------------------------
class Mouth:
    """Vocal output organ for Atlas.

    Attempts Kokoro TTS synthesis; gracefully falls back to console output
    if Kokoro is uninstalled, unconfigured, or encounters an error.
    """

    def speak(self, text: str) -> None:
        """Synthesizes text using the Kokoro TTS model and plays the audio."""
        print(f"[Mouth]: {text}", flush=True)

        # import pyttsx3

        # # 1. Initialize the TTS engine
        # engine = pyttsx3.init()

        # # 2. Adjust Speech Properties (Optional)
        # # Speed (default is usually around 200)
        # rate = engine.getProperty('rate')
        # engine.setProperty('rate', 150)  # Slow it down a bit

        # # Volume (0.0 min to 1.0 max)
        # volume = engine.getProperty('volume')
        # engine.setProperty('volume', 0.9)

        # # 3. Change Voice (Optional)
        # # 0 is male
        # voices = list(cast(Any, engine.getProperty("voices")) or [])
        # if voices:
        #     engine.setProperty("voice", voices[0].id)

        # # 4. Speak the Text
        # engine.say(text)

        # # 5. Process the cue and block until finished
        # engine.runAndWait()

        from contextlib import redirect_stdout
        from contextlib import redirect_stderr
        from io import StringIO

        chunks: list[np.ndarray] = []
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            from kokoro import KPipeline

            pipeline = KPipeline(
                lang_code="b",
                repo_id="hexgrad/Kokoro-82M",
                device="cuda",
            )

            for _, _, audio in pipeline(text, voice="bm_george", speed=1.0):
                audio_value = cast(Any, audio)
                if audio_value is None or isinstance(audio_value, str):
                    continue
                if hasattr(audio_value, "cpu"):
                    audio_value = audio_value.cpu().numpy()
                chunks.append(np.asarray(audio_value))

        if chunks:
            sd.play(np.concatenate(chunks), samplerate=24000)
            sd.wait()