"""Fast startup check for PROJECT INVISIBLE - Ideogram 4.

Forge runs this file in a separate Python process on every launch. Importing
Torch, Transformers, CUDA or bitsandbytes here duplicated work done by Forge
and added several seconds to startup. The normal extension engine performs its
hardware/configuration check after Forge has loaded those libraries.
"""

from __future__ import annotations

import importlib.util


TAG = "[Invisible-I4]"
REQUIRED = ("torch", "transformers", "safetensors", "einops", "huggingface_hub", "tqdm")


def main() -> None:
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    if missing:
        print(f"{TAG} missing Forge packages: {', '.join(missing)}")
        return

    print(f"{TAG} startup check OK (hardware is checked by the loaded extension)")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"{TAG} startup check skipped (non-fatal): {error}")
