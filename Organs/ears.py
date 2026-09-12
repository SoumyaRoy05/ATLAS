import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel

# Make the project root available when this file is run directly.
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Organs.brain import Brain

#Settings for the Whisper model
sample_rate = 16000 # Sample rate is the number of samples per second, in Hz
block_duration = 0.5 # Block duration is the time interval between each audio block, in seconds
chunk_duration = 5 # Chunk duration is the time interval for each audio chunk, in seconds
channels = 1 # mono audio

frames_per_block = int(sample_rate * block_duration)
frames_per_chunk = int(sample_rate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []
speech_texts = []


# Model Setup
model_size = "small"  # Use a smaller model for faster inference

print("[Ears]: Loading Whisper model...", flush=True)
model = WhisperModel(model_size, device="cuda", compute_type="float32")

# used to capture audio from the microphone and put it into a queue for processing
def audio_callback(indata, frames, time, status):
    """Callback function to capture audio from the microphone."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

# used for continuously recording audio and storing it in a buffer for processing
def recorder():
    """Continuously records audio and stores it in a buffer."""
    with sd.InputStream(samplerate=sample_rate, channels=channels,
                        blocksize=frames_per_block, callback=audio_callback):
        print("[Ears]: Listening...", flush=True)

        while True:
            sd.sleep(100)

# used for transcribing the audio in the buffer using the Whisper model
def transcriber():
    """Transcribes the audio in the buffer."""

    global audio_buffer
    global speech_texts


    while True:
        block = audio_queue.get()
        audio_buffer.append(block)

        total_frames = sum(len(b) for b in audio_buffer)
        # rto check if we have enough frames to process a chunk
        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            audio_buffer = [] # clear buffer after processing

            audio_data = audio_data.flatten().astype(np.float32)  # Flatten and convert to float32

            segments, _ = model.transcribe(
                audio_data,
                language="en",
                beam_size=1,
            )

            transcribed = False
            for segment in segments:
                transcribed = True
                print(f"[Ears]: {segment.text}", flush=True)
                speech_texts.append(segment.text.strip())

            if not transcribed:
                print(f"[Ears]: Prompt well recieved, Sir!", flush=True)
                Brain().think(" ".join(speech_texts))
                speech_texts = []  # Clear the list after speaking  
                break


def hear():
    print(f"[Ears]: At your command, Boss!", flush=True)
    threading.Thread(target=recorder, daemon=True).start()
    transcriber()