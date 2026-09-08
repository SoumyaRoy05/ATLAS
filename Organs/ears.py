import os
import sys
import queue
import time
import warnings
import numpy as np
from collections import deque
import sounddevice as sd
from typing import Optional
from faster_whisper import WhisperModel
from pathlib import Path
from openwakeword.model import Model
from openwakeword import models as openwakeword_models


# ------------------------------------------------------------------------------
# Hardware & Model Path Setup
# ------------------------------------------------------------------------------
nvidia_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
if nvidia_dir.exists():
    for bin_folder in nvidia_dir.rglob("bin"):
        if bin_folder.is_dir():
            os.add_dll_directory(str(bin_folder))
            os.environ["PATH"] = str(bin_folder) + os.pathsep + os.environ.get("PATH", "")

if not os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = r"D:\Codes\AI_MODELS\huggingface"


# -------------------------------------------------------------------------------
# Connect to the Brain module, which handles higher-level processing of transcribed speech.
# -------------------------------------------------------------------------------
try:
    from Organs.brain import Brain
except ImportError:
    from brain import Brain

# -------------------------------------------------------------------------------
# Ears Class: Listens for wake stimuli, captures speech, and sends transcribed words to the brain.
# -------------------------------------------------------------------------------
class Ears:
    """Sensory Organ: Listens for wake stimuli, captures speech,

    and sends transcribed words to the brain.
    """

    # uses sounddevice to capture audio, openwakeword to detect wake words, and faster-whisper to transcribe speech
    def __init__(
        self,
        wake_threshold: float = 0.6,
        energy_threshold: float = 450.0,
        silence_timeout_sec: float = 2.0,
        speech_start_timeout_sec: float = 2.0,
        command_timeout_sec: float = 30.0,
        conversation_timeout_sec: float = 15.0,
        partial_interval_sec: float = 0.5,
        rolling_window_sec: float = 4.0,
    ):
        self.sample_rate = 16000
        self.chunk_size = 1280  # 80 ms audio frames
        self.wake_threshold = wake_threshold
        self.energy_threshold = energy_threshold
        self.silence_timeout_sec = silence_timeout_sec
        self.silence_frames = int(silence_timeout_sec * self.sample_rate / self.chunk_size)
        self.speech_start_timeout_sec = speech_start_timeout_sec
        self.command_timeout_sec = command_timeout_sec
        self.conversation_timeout_sec = conversation_timeout_sec
        self.partial_interval_sec = partial_interval_sec
        self.rolling_window_sec = rolling_window_sec

        self.audio_queue = queue.Queue()
        self.is_running = False

        # 3 Steps of the Ears organ:
        # 1. Auditory Gatekeeper (openWakeWord on CPU): Detects wake words in real-time audio streams.
        model_dir = Path(r"D:\Codes\AI_MODELS\openwakeword")
        models = [str(f) for f in model_dir.glob("*.onnx") if "embedding" not in f.name and "melspec" not in f.name] if model_dir.exists() else []
        if not models:
            hey_atlas = openwakeword_models.get("hey_atlas")
            if hey_atlas is None:
                raise FileNotFoundError(
                    f"The hey_atlas wake-word model was not found in {model_dir}. "
                    "Place hey_atlas.onnx there before starting Atlas."
                )
            models = [hey_atlas["model_path"]]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Specified provider 'CUDAExecutionProvider' is not in available provider names.*",
            )
            self.oww = Model(wakeword_model_paths=models)

        # 2. Auditory Transduction / STT (faster-whisper on GPU): Transcribes speech into text.
        self.stt = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
        self.cpu_stt = None

        # 3. Synaptic Connection to Brain: Transmits transcribed text to the Brain module for higher-level processing.
        self.brain = Brain()


    def stop(self) -> None:
        """Stops audio capture and releases any blocked listening operation."""
        self.is_running = False
        sd.stop()
        self.audio_queue.put(np.zeros(self.chunk_size, dtype=np.int16))


    def _reset_listening_state(self) -> None:
        """Clears the previous exchange before waiting for the next wake word."""
        self.oww.reset()
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break


    # Audio callback function for the sounddevice input stream; puts audio chunks into a queue for processing
    def _audio_callback(self, indata, frames, time_info, status):
        self.audio_queue.put(indata.copy())


    def _capture_utterance(self, speech_start_timeout_sec: float) -> str:
        """Stream partial Whisper text and return one finalized utterance."""
        frames = []
        rolling_frames = deque(
            maxlen=max(1, int(self.rolling_window_sec * self.sample_rate / self.chunk_size))
        )
        speech_started = False
        started_at = time.monotonic()
        last_voice_at = started_at
        last_partial_at = started_at
        last_partial = ""
        speech_notice_shown = False

        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=0.25).flatten()
            except queue.Empty:
                continue

            frames.append(chunk)
            rolling_frames.append(chunk)

            elapsed = time.monotonic() - started_at
            rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))

            if rms > self.energy_threshold:
                speech_started = True
                last_voice_at = time.monotonic()
                if not speech_notice_shown:
                    print("\r\033[2K[You]: listening...", end="", flush=True)
                    speech_notice_shown = True
                if elapsed - last_partial_at >= self.partial_interval_sec and len(rolling_frames) >= 2:
                    partial = self.transcribe(list(rolling_frames), vad_filter=True)
                    if partial and partial != last_partial:
                        print(f"\r\033[2K[You]: {partial}", end="", flush=True)
                        last_partial = partial
                    last_partial_at = elapsed

            if not speech_started and elapsed >= speech_start_timeout_sec:
                break
            if speech_started and time.monotonic() - last_voice_at >= self.silence_timeout_sec:
                break
            if elapsed >= self.command_timeout_sec:
                break

        final_text = self.transcribe(frames, vad_filter=True)
        if final_text:
            print(f"\r\033[2K[You]: {final_text}", flush=True)
        return final_text


    # Transcribes buffered audio frames into text using the Whisper model.
    def transcribe(self, frames: list, vad_filter: bool = True) -> str:
        """Converts raw sound waves into plain-text words."""
        if not frames:
            return ""
        audio = np.concatenate(frames, axis=0).astype(np.float32) / 32768.0
        try:
            segments, _ = self.stt.transcribe(
                audio,
                beam_size=1,
                language="en",
                vad_filter=vad_filter,
                vad_parameters={"min_silence_duration_ms": 300},
                condition_on_previous_text=False,
                temperature=0.0,
                without_timestamps=True,
            )
        except RuntimeError as error:
            if "cublas" not in str(error).lower():
                raise
            if self.cpu_stt is None:
                print("\n[Ears]: CUDA speech recognition unavailable. Switching to CPU.", flush=True)
                self.cpu_stt = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
            segments, _ = self.cpu_stt.transcribe(
                audio,
                beam_size=1,
                language="en",
                vad_filter=vad_filter,
                vad_parameters={"min_silence_duration_ms": 300},
                condition_on_previous_text=False,
                temperature=0.0,
                without_timestamps=True,
            )
        return " ".join(seg.text.strip() for seg in segments).strip()


    # Starts the main listening loop for capturing and processing audio.
    def start(self):
        """Main listening loop: Wake -> Record -> Transcribe -> Send to Brain."""
        self.is_running = True
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            channels=1,
            dtype="int16",
            callback=self._audio_callback,
        )

        with stream:
            print("[Ears]: Listening...")
            while self.is_running:
                frame = self.audio_queue.get().flatten()

                # Step 1: Detect wake word
                prediction = self.oww.predict(frame)
                scores = prediction[0] if isinstance(prediction, tuple) else prediction
                if any(score >= self.wake_threshold for score in scores.values()):
                    print("\n[Ears]: Stimulus perceived. Listening to command...")
                    self.oww.reset()

                    # Step 2: Stream and finalize the question after five seconds of silence.
                    query = self._capture_utterance(self.speech_start_timeout_sec)
                    if query:
                        # Step 3: Process the question immediately after the pause.
                        self.brain.think(query)

                        # Step 4: Keep the conversation open for follow-ups after Mouth speaks.
                        while self.is_running:
                            self._reset_listening_state()
                            print(
                                f"[Ears]: Response complete. Listening for a follow-up for {self.conversation_timeout_sec:.0f} seconds...",
                                flush=True,
                            )
                            follow_up = self._capture_utterance(self.conversation_timeout_sec)
                            if not follow_up:
                                break
                            self.brain.think(follow_up)

                    self._reset_listening_state()
                    print("\n[Ears]: Returning to passive listening...")


if __name__ == "__main__":
    ears = Ears()
    try:
        ears.start()
    except KeyboardInterrupt:
        ears.is_running = False
        print("\n[Ears]: Offline.")