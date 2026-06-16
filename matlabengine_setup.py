import glob
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple


MATLAB_RELEASE_PATTERN = re.compile(r"R?(20\d{2}[ab])")
MATLAB_ROOT_ENV_VARS = ("ULTRALYZER_MATLABROOT", "MATLABROOT")
MATLAB_ENGINE_SPECS = {
    "R2020b": {"version": "9.9.6", "requires_python": ">=3.6,<3.9"},
    "R2021a": {"version": "9.10.5", "requires_python": ">=3.7,<3.9"},
    "R2021b": {"version": "9.11.23", "requires_python": ">=3.7,<3.10"},
    "R2022a": {"version": "9.12.21", "requires_python": ">=3.8,<3.10"},
    "R2022b": {"version": "9.13.11", "requires_python": ">=3.8,<3.11"},
    "R2023a": {"version": "9.14.7", "requires_python": ">=3.8,<3.11"},
    "R2023b": {"version": "23.2.3", "requires_python": ">=3.9,<3.12"},
    "R2024a": {"version": "24.1.4", "requires_python": ">=3.9,<3.12"},
    "R2024b": {"version": "24.2.2", "requires_python": ">=3.9,<3.13"},
    "R2025a": {"version": "25.1.2", "requires_python": ">=3.9,<3.13"},
    "R2025b": {"version": "25.2.2", "requires_python": ">=3.9,<3.13"},
    "R2026a": {"version": "26.1.12", "requires_python": ">=3.9,<3.14"},
}
NONDEFAULT_LIBRARY_PATH_HINTS = {
    "darwin": ("DYLD_LIBRARY_PATH", "maci64"),
    "linux": ("LD_LIBRARY_PATH", "glnxa64"),
}


def normalize_matlab_release(value: str) -> Optional[str]:
    match = MATLAB_RELEASE_PATTERN.search(value)
    if match is None:
        return None
    return "R{0}".format(match.group(1))


def get_matlab_engine_spec(release: str) -> Optional[Dict[str, str]]:
    return MATLAB_ENGINE_SPECS.get(release)


def python_version_text(version_info: Optional[Sequence[int]] = None) -> str:
    if version_info is None:
        version_info = sys.version_info[:3]
    return ".".join(str(part) for part in version_info[:3])


def is_python_compatible(
    requires_python: str,
    version_info: Optional[Sequence[int]] = None,
) -> bool:
    current = _normalize_version_tuple(version_info or sys.version_info[:3])
    clauses = [clause.strip() for clause in requires_python.split(",") if clause.strip()]

    for clause in clauses:
        operator = None
        for candidate in (">=", "<=", "==", ">", "<"):
            if clause.startswith(candidate):
                operator = candidate
                break

        if operator is None:
            raise ValueError("Unsupported Python specifier: {0}".format(clause))

        target = _normalize_version_tuple(_parse_version_text(clause[len(operator) :]))

        if operator == ">=" and current < target:
            return False
        if operator == ">" and current <= target:
            return False
        if operator == "<=" and current > target:
            return False
        if operator == "<" and current >= target:
            return False
        if operator == "==" and current != target:
            return False

    return True


def detect_matlab_release_from_path(candidate: str) -> Optional[Tuple[str, str]]:
    path = Path(os.path.expanduser(candidate))
    if not path.exists():
        return None

    try:
        resolved = path.resolve()
    except OSError:
        resolved = path

    for current in [resolved] + list(resolved.parents):
        release = normalize_matlab_release(current.name)
        if release is not None:
            return str(current), release

    matlab_command = _matlab_command_from_path(resolved)
    if matlab_command is None:
        return None

    release = _query_matlab_release(matlab_command)
    if release is None:
        return None

    root = _matlab_root_from_command(matlab_command)
    return root, release


def detect_matlab_installation() -> Optional[Tuple[str, str]]:
    for env_name in MATLAB_ROOT_ENV_VARS:
        candidate = os.environ.get(env_name)
        if not candidate:
            continue

        detected = detect_matlab_release_from_path(candidate)
        if detected is not None:
            return detected

        print(
            "Could not determine the MATLAB release from {0}={1}.".format(
                env_name, candidate
            )
        )

    matlab_command = shutil.which("matlab")
    if matlab_command:
        detected = detect_matlab_release_from_path(matlab_command)
        if detected is not None:
            return detected

    candidates = {}
    for candidate in iter_standard_matlab_roots():
        detected = detect_matlab_release_from_path(candidate)
        if detected is None:
            continue
        root, release = detected
        candidates[root] = release

    if not candidates:
        return None

    root, release = max(
        candidates.items(),
        key=lambda item: _release_sort_key(item[1]),
    )
    return root, release


def iter_standard_matlab_roots() -> Iterable[str]:
    system = platform.system().lower()
    patterns = []

    if system == "windows":
        for env_name in ("ProgramW6432", "ProgramFiles"):
            base = os.environ.get(env_name)
            if base:
                patterns.append(os.path.join(base, "MATLAB", "R20*[ab]"))
        patterns.append(r"C:\Program Files\MATLAB\R20*[ab]")
    elif system == "darwin":
        patterns.append("/Applications/MATLAB_R20*[ab].app")
    elif system == "linux":
        patterns.append("/usr/local/MATLAB/R20*[ab]")

    for pattern in patterns:
        for candidate in sorted(glob.glob(pattern)):
            if os.path.exists(candidate):
                yield candidate


def install_detected_matlabengine() -> None:
    detected = detect_matlab_installation()
    if detected is None:
        print("MATLAB installation not found. Skipping matlabengine installation.")
        print(
            "Set ULTRALYZER_MATLABROOT to your MATLAB root if MATLAB is installed in a non-standard location."
        )
        return

    matlab_root, release = detected
    spec = get_matlab_engine_spec(release)
    if spec is None:
        print(
            "Detected MATLAB {0} at {1}, but this installer does not know the matching matlabengine package version.".format(
                release, matlab_root
            )
        )
        print(
            "Install matlabengine manually after checking the MATLAB Engine API for Python release history on PyPI."
        )
        return

    version = spec["version"]
    requires_python = spec["requires_python"]

    if not is_python_compatible(requires_python):
        print(
            "Detected MATLAB {0} at {1}, but matlabengine=={2} requires Python {3}.".format(
                release, matlab_root, version, requires_python
            )
        )
        print(
            "Current interpreter is Python {0}. Skipping matlabengine installation.".format(
                python_version_text()
            )
        )
        return

    _print_library_path_hint(matlab_root, release)

    package = "matlabengine=={0}".format(version)
    print("Detected MATLAB {0} at {1}.".format(release, matlab_root))
    print("Installing {0}...".format(package))
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("MATLAB engine installed successfully.")


def _default_matlab_root(system: str, release: str) -> Optional[str]:
    if system == "darwin":
        return "/Applications/MATLAB_{0}.app".format(release)
    if system == "linux":
        return "/usr/local/MATLAB/{0}".format(release)
    return None


def _matlab_command_from_path(path: Path) -> Optional[str]:
    command_name = "matlab.exe" if platform.system().lower() == "windows" else "matlab"

    if path.is_file() and path.name.lower().startswith("matlab"):
        return str(path)

    if path.is_dir():
        command = path / "bin" / command_name
        if command.exists():
            return str(command)

        if path.name.lower() == "bin":
            command = path / command_name
            if command.exists():
                return str(command)

    return None


def _matlab_root_from_command(command_path: str) -> str:
    command = Path(command_path)
    if command.parent.name.lower() == "bin":
        return str(command.parent.parent)
    return str(command.parent)


def _normalize_version_tuple(version_info: Sequence[int]) -> Tuple[int, int, int]:
    normalized = list(version_info[:3])
    while len(normalized) < 3:
        normalized.append(0)
    return tuple(int(part) for part in normalized[:3])


def _parse_version_text(version_text: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in version_text.split(".") if part)


def _print_library_path_hint(matlab_root: str, release: str) -> None:
    system = platform.system().lower()
    hint = NONDEFAULT_LIBRARY_PATH_HINTS.get(system)
    if hint is None:
        return

    default_root = _default_matlab_root(system, release)
    if default_root is None:
        return

    if os.path.normcase(os.path.abspath(matlab_root)) == os.path.normcase(
        os.path.abspath(default_root)
    ):
        return

    env_var, arch_dir = hint
    library_dir = os.path.join(matlab_root, "bin", arch_dir)
    print("MATLAB was detected in a non-default location.")
    print(
        "If importing matlab.engine fails later, add {0} to {1}.".format(
            library_dir, env_var
        )
    )


def _query_matlab_release(command_path: str) -> Optional[str]:
    try:
        result = subprocess.run(
            [
                command_path,
                "-batch",
                "fprintf('ULTRALYZER_MATLAB_RELEASE=%s\\n', version('-release'))",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return normalize_matlab_release(output)


def _release_sort_key(release: str) -> Tuple[int, int]:
    return int(release[1:5]), 0 if release.endswith("a") else 1