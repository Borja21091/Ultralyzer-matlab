import os
import subprocess
import sys

try:
    from matlabengine_setup import install_detected_matlabengine
except ImportError:
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from matlabengine_setup import install_detected_matlabengine


def install_requirements() -> None:
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        print("Installing dependencies from requirements.txt...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            print("Requirements installed successfully.")
        except subprocess.CalledProcessError:
            print("Failed to install requirements.")
            sys.exit(1)
    else:
        print(f"{req_file} not found. Skipping.")


def main() -> None:
    install_requirements()
    install_detected_matlabengine()
    print("Setup completed successfully!")
    print("Set ULTRALYZER_GEOMETRY_BACKENDS=matlab before launching Ultralyzer.")
    print("If the MATLAB geometry backend is unavailable, the app still runs but some metrics may be skipped.")


if __name__ == "__main__":
    main()