import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. Hardware Acceleration & Dynamic Environment Setup
# -----------------------------------------------------------------------------
# Inject NVIDIA CUDA and cuBLAS DLL binaries into the Windows runtime path
nvidia_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
if nvidia_dir.exists():
    for bin_folder in nvidia_dir.rglob("bin"):
        if bin_folder.is_dir():
            os.add_dll_directory(str(bin_folder))
            os.environ["PATH"] = str(bin_folder) + os.pathsep + os.environ.get("PATH", "")

# Locate and load environment variables from the project root .env
root_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=root_dir / ".env")

# Guarantee Hugging Face models route to the dedicated drive cache
if not os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = r"D:\Codes\AI_MODELS\huggingface"


# -----------------------------------------------------------------------------
# 2. Sensory Organ Import
# -----------------------------------------------------------------------------
from Organs.ears import hear  # Auditory sensory organ: wake word detection + speech-to-text


# -----------------------------------------------------------------------------
# 3. Central Nervous System Lifecycle Supervisor
# -----------------------------------------------------------------------------
def main() -> None:
    """Master ignition routine for Atlas.

    Initializes hardware contexts, boots the acoustic sensory organ,
    and supervises the event loop until clean termination.
    """
    print("=" * 60)
    print("               ATLAS DIGITAL STEWARD ONLINE                 ")
    print("=" * 60)
    print("[CNS] Initializing acoustic and cognitive subsystems...")

    ears = None

    try:
        # Instantiates Ears, which internally connects to Brain and Mouth
        ears = hear()
        print("[CNS] Ignition sequence complete. Acoustic sensory organ active.")
        # Starts the uninterrupted listening loop:
        # Ears (OWW + Whisper) -> Brain (LangGraph + LLM) -> Mouth (Kokoro/TTS)
        while True:
            time.sleep(1)  # Keep the supervisor alive without constantly consuming CPU

    except KeyboardInterrupt:
        print("\n[CNS] Manual keyboard interrupt received. Commencing safe teardown...")

    except SystemExit:
        print("\n[CNS] System power-down command confirmed. Standing down...")

    except Exception as e:
        print(f"\n[CNS - Critical Fault]: Unhandled pipeline exception: {e}")

    finally:
        # Guarantee audio buffers, PortAudio streams, and GPU allocations release cleanly
        stop = getattr(ears, "stop", None) if ears is not None else None
        if callable(stop):
            stop()
        print("[CNS] Audio hardware unhooked. GPU contexts released. System offline.")



if __name__ == "__main__":
    main()