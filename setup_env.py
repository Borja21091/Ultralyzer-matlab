import os
import subprocess
import sys


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
    print("Setup completed successfully!")
    print("Set ULTRALYZER_GEOMETRY_BACKENDS=matlab before launching Ultralyzer.")
    print("If the MATLAB geometry backend is unavailable, the app still runs but some metrics may be skipped.")


if __name__ == "__main__":
    main()