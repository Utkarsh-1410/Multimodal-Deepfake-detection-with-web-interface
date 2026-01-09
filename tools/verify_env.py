from __future__ import annotations

import importlib
import sys


REQUIRED_MODULES = [
    # Web
    "fastapi",
    "uvicorn",
    "jinja2",
    "multipart",
    # Core
    "yaml",
    "numpy",
    "pandas",
    "PIL",
    "cv2",
    # ML
    "torch",
    "torchvision",
    "timm",
    "facenet_pytorch",
    "skimage",
    "sklearn",
    # Utils
    "tqdm",
    "tabulate",
    "gdown",
    "imutils",
]

OPTIONAL_MODULES = [
    # Optional (used by some pipelines/features, but not required for basic web inference)
    "torchaudio",
]


def _try_import(name: str) -> str | None:
    try:
        importlib.import_module(name)
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{name}: {exc}"


def main() -> int:
    failures: list[str] = []
    optional_failures: list[str] = []

    for module_name in REQUIRED_MODULES:
        err = _try_import(module_name)
        if err:
            failures.append(err)

    for module_name in OPTIONAL_MODULES:
        err = _try_import(module_name)
        if err:
            optional_failures.append(err)

    if failures:
        print("Missing/broken imports detected:\n")
        for line in failures:
            print(f"- {line}")
        print("\nInstall runtime deps with:\n  pip install -r requirements-runtime.txt\n  pip install -r web/requirements.txt\n")
        return 1

    if optional_failures:
        print("Optional imports missing (safe to ignore for basic inference):\n")
        for line in optional_failures:
            print(f"- {line}")
        print("")

    print("Environment looks OK (imports succeeded).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
