import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. Hardware Acceleration & Environment Setup
# -----------------------------------------------------------------------------
# Register NVIDIA CUDA/cuBLAS DLL paths into Windows runtime
nvidia_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
if nvidia_dir.exists():
    for bin_folder in nvidia_dir.rglob("bin"):
        if bin_folder.is_dir():
            os.add_dll_directory(str(bin_folder))
            os.environ["PATH"] = str(bin_folder) + os.pathsep + os.environ.get("PATH", "")

# Load configuration secrets (.env)
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# Enforce Hugging Face cache destination
hf_home = os.getenv("HF_HOME")
if hf_home is not None:
    os.environ["HF_HOME"] = hf_home


# -----------------------------------------------------------------------------
# 2. Sensory Organ Initialization
# -----------------------------------------------------------------------------
try:
    from Organs.ears import Ears
except ImportError:
    from Organs.ears import Ears  # type: ignore


# -----------------------------------------------------------------------------
# 3. Lifecycle Supervisor & Ignition
# -----------------------------------------------------------------------------
def main() -> None:
    print("=== Atlas System Online ===")
    print("[CNS] Igniting sensory organ: Ears...")

    # Initialize the acoustic organ
    ears = Ears()

    try:
        # Starts the autonomous organ chain:
        # ears.py -> take_input() -> brain.py (calls llm.py) -> mouth.speak()
        ears.start()

    except KeyboardInterrupt:
        print("\n[CNS] Manual keyboard interrupt detected. Halting pipeline...")

    except SystemExit:
        print("\n[CNS] System termination command verified. Standing down...")

    finally:
        # Ensure microphone streams, buffers, and background threads close cleanly
        if hasattr(ears, "stop"):
            ears.stop()
        print("[CNS] Audio streams released. Hardware deallocated. System offline.")


if __name__ == "__main__":
    main()