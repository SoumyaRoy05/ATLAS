import os
import sys
import queue
import time
from pathlib import Path
from typing import cast, Optional
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from openwakeword import models as openwakeword_models
from openwakeword.model import Model




# -----------------------------------------------------------------------------
# 1. Hardware Acceleration & Dynamic Environment Setup
# -----------------------------------------------------------------------------
# Register NVIDIA CUDA/cuBLAS DLL paths into Windows runtime
nvidia_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
if nvidia_dir.exists():
    for bin_folder in nvidia_dir.rglob("bin"):
        if bin_folder.is_dir():
            os.add_dll_directory(str(bin_folder))
            os.environ["PATH"] = str(bin_folder) + os.pathsep + os.environ.get("PATH", "")

# Centralized Hugging Face cache directory
if not os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = r"D:\Codes\AI_MODELS\huggingface"




# -----------------------------------------------------------------------------
# 2. Downstream Organ & Behaviour Imports
# -----------------------------------------------------------------------------
try:
    from Organs.brain import Brain
    from Organs.mouth import Mouth
    from Behaviour.persona import get_wake_receipt, get_dismiss_receipt
except ImportError:
    try:
        from brain import Brain
        from mouth import Mouth
        from Behaviour.persona import get_wake_receipt, get_dismiss_receipt
    except ImportError:
        Brain = None
        Mouth = None
        get_wake_receipt = lambda: "At your command."
        get_dismiss_receipt = lambda: "Standing down."




class Ears:
    """Acoustic sensory organ: handles microphone capture, wake-word detection,

    barge-in monitoring, and speech transcription via faster-whisper.
    """

    # function for initializing the Ears class with default parameters for wake-word detection and speech transcription
    def __init__(
        self,
        wake_threshold: float = 0.6,
        energy_threshold: float = 450.0,
        active_timeout_sec: float = 10.0,
        silence_timeout_sec: float = 1.4,
        initial_timeout_sec: float = 2.5,
        max_record_sec: float = 12.0,
    ):
        self.sample_rate = 16000
        self.chunk_size = 1280  # Strict 80 ms audio frames (16 kHz int16)
        self.wake_threshold = wake_threshold
        self.energy_threshold = energy_threshold
        self.active_timeout_sec = active_timeout_sec

        # Calculate frame boundaries
        self.silence_frames_needed = int(silence_timeout_sec * self.sample_rate / self.chunk_size)
        self.initial_timeout_frames = int(initial_timeout_sec * self.sample_rate / self.chunk_size)
        self.max_record_frames = int(max_record_sec * self.sample_rate / self.chunk_size)

        # Thread-safe audio buffer
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.stream: Optional[sd.InputStream] = None

        # Display active microphone
        raw_mic = sd.query_devices(kind="input")
        default_mic = dict(raw_mic) if isinstance(raw_mic, dict) else {}
        mic_name = default_mic.get("name", "Default System Microphone")
        mic_idx = default_mic.get("index", sd.default.device[0])
        print(f"[Ears] Your current microphone is: [{mic_idx}] {mic_name}")

        # Initialize openWakeWord on CPU
        print("[Ears] Initializing openWakeWord models...")
        custom_model_dir = Path(r"D:\Codes\AI_MODELS\OpenWakeWord")
        loaded_model_paths = [] # List to hold paths of loaded wake-word models

        # Check for custom models or load the bundled fallback
        for name in ["atlas", "hey_atlas", "buddy", "pal"]:
            custom_path = custom_model_dir / f"{name}.onnx"
            if custom_path.exists():
                loaded_model_paths.append(str(custom_path))

        # If no custom models are found, default to the bundled hey_jarvis model
        if not loaded_model_paths:
            # Default to bundled hey_jarvis model if custom wake models are not yet trained
            loaded_model_paths = [openwakeword_models["hey_jarvis"]["model_path"]]

        self.oww_model = Model(wakeword_model_paths=loaded_model_paths)

        # Initialize faster-whisper on GPU
        print("[Ears] Loading faster-whisper (large-v3-turbo) onto GPU...")
        self.stt_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")

        # Instantiate downstream organs
        self.brain = Brain() if Brain else None
        self.mouth = Mouth() if Mouth else None

        # Hardcoded primitive command filters
        self.dismissal_phrases = ["dismissed", "stand down", "that will be all", "sleep", "go to sleep"]
        self.shutdown_phrases = ["shutdown", "shut down", "power off", "terminate system"]


    # function for handling incoming audio frames from the microphone and placing them into a queue for processing
    def _audio_callback(self, indata, frames, time_info, status):
        """Streams raw 16-bit PCM frames into the queue."""
        self.audio_queue.put(indata.copy())


    # function for clearing any residual audio frames from the queue to prevent processing stale data
    def _flush_queue(self):
        """Clears residual background noise from the queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break


    # function for converting accumulated audio frames into a format suitable for transcription
    # and then using the faster-whisper model to trans
    def take_input(self, audio_chunks: list) -> str:
        """Converts accumulated int16 audio frames into a normalized float32 array

        and transcribes the utterance using faster-whisper.
        """
        print("[Ears] Transcribing speech via faster-whisper...")
        audio_int16 = np.concatenate(audio_chunks, axis=0)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        segments, _ = self.stt_model.transcribe(
            audio_float32,
            beam_size=1,
            language="en",
            vad_filter=True
        )
        query = " ".join([seg.text.strip() for seg in segments]).strip()
        return query

    # function for starting the main listening loop, which handles wake-word detection, speech recording, and downstream processing
    def start(self):
        """Starts the main acoustic listening loop and orchestrates downstream handoff."""
        self.is_running = True
        state = "IDLE"  # "IDLE" (Wake Spotting) or "ACTIVE" (Conversational Window)
        last_active_time = 0.0

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            channels=1,
            dtype="int16",
            callback=self._audio_callback
        )

        with self.stream:
            print("\n[IDLE] Monitoring for wake triggers ('Atlas', 'Hey Atlas', 'Buddy', 'Pal')...")

            while self.is_running:
                frame = self.audio_queue.get().flatten()


                # -------------------------------------------------------------
                # Active Barge-In Interception (While Mouth Is Speaking)
                # -------------------------------------------------------------

                if self.mouth and getattr(self.mouth, "is_speaking", False):
                    barge_pred = cast(dict[str, float], self.oww_model.predict(frame)) # Predict wake word in the presence of ongoing speech

                    # If any wake word is detected above the threshold, halt playback and reset the model
                    if any(score >= self.wake_threshold for score in barge_pred.values()):
                        print("\n[Barge-In] Wake word detected during speech. Halting playback...")

                        # If the mouth organ has a stop method, call it to halt speech output
                        if hasattr(self.mouth, "stop"):
                            self.mouth.stop()
                        self.oww_model.reset()
                        self._flush_queue()
                        state = "ACTIVE"
                        last_active_time = time.time()
                        continue

                # -------------------------------------------------------------
                # State 1: IDLE - Low-Power Acoustic Wake Spotting
                # -------------------------------------------------------------
                if state == "IDLE":
                    prediction = cast(dict[str, float], self.oww_model.predict(frame))
                    max_score = max(prediction.values()) if prediction else 0.0

                    if max_score >= self.wake_threshold:
                        print(f"\n[ACTIVE] Wake trigger verified ({max_score:.2f}). Session engaged.")
                        self.oww_model.reset()
                        self._flush_queue()

                        # Optional acoustic wake confirmation
                        if self.mouth and hasattr(self.mouth, "speak"):
                            self.mouth.speak(get_wake_receipt())

                        state = "ACTIVE"
                        last_active_time = time.time()
                    continue

                # -------------------------------------------------------------
                # State 2: ACTIVE - Conversational Window & Utterance Recording
                # -------------------------------------------------------------
                elif state == "ACTIVE":
                    # Check if conversational window elapsed
                    if time.time() - last_active_time > self.active_timeout_sec:
                        print("\n[IDLE] Attention window elapsed. Reverting to low-power wake listener...")
                        state = "IDLE"
                        self.oww_model.reset()
                        self._flush_queue()
                        continue

                    # Record incoming speech
                    recorded_chunks = [frame]
                    speech_detected = False
                    silence_frames = 0
                    waiting_frames = 0

                    while self.is_running:
                        voice_frame = self.audio_queue.get().flatten()
                        recorded_chunks.append(voice_frame)

                        # RMS energy boundary detection
                        rms = np.sqrt(np.mean(voice_frame.astype(np.float64) ** 2))

                        if rms > self.energy_threshold:
                            speech_detected = True
                            silence_frames = 0
                        else:
                            if speech_detected:
                                silence_frames += 1
                                if silence_frames >= self.silence_frames_needed:
                                    break  # User finished speaking
                            else:
                                waiting_frames += 1
                                if waiting_frames >= self.initial_timeout_frames:
                                    break  # User didn't speak within initial timeout

                        if len(recorded_chunks) >= self.max_record_frames:
                            break

                    # If no speech initiated, continue listening inside the active window
                    if not speech_detected:
                        continue

                    # Transcribe captured audio via take_input()
                    user_prompt = self.take_input(recorded_chunks)

                    if not user_prompt:
                        continue

                    print(f"\n[User]: {user_prompt}")
                    lower_prompt = user_prompt.lower()

                    # ---------------------------------------------------------
                    # Primitive Reflex & Termination Checks
                    # ---------------------------------------------------------
                    # Hardcoded system termination
                    if any(phrase in lower_prompt for phrase in self.shutdown_phrases):
                        if self.mouth:
                            self.mouth.speak("Deactivating all systems. Standing down completely.")
                        self.stop()
                        raise SystemExit(0)

                    # Hardcoded dismissal reflex
                    if any(phrase in lower_prompt for phrase in self.dismissal_phrases):
                        if self.mouth:
                            self.mouth.speak(get_dismiss_receipt())
                        print("\n[IDLE] Dismissal receipt acknowledged. Standby mode.")
                        state = "IDLE"
                        self._flush_queue()
                        continue

                    # ---------------------------------------------------------
                    # Downstream Pipeline Relay: take_input() -> Brain -> Mouth
                    # ---------------------------------------------------------
                    if self.brain and hasattr(self.brain, "think"):
                        assistant_reply = self.brain.think(user_prompt)
                    else:
                        assistant_reply = f"Cognitive core offline. Transcribed: {user_prompt}"

                    if self.mouth and hasattr(self.mouth, "speak"):
                        self.mouth.speak(assistant_reply)
                    else:
                        print(f"[Atlas]: {assistant_reply}")

                    # Reset active timer after reply finishes
                    last_active_time = time.time()
                    self._flush_queue()

    def stop(self):
        """Safely stops audio streams and releases device resources."""
        self.is_running = False
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()
        self._flush_queue()