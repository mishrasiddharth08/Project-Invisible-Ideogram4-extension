"""PROJECT INVISIBLE - engine-namespaced asset subsystem for Ideogram 4.

- huggingface_hub (already in the Forge venv): resume + etag verify + tqdm
  progress to stderr (visible in the DOS window).
- Civitai public model API for the gray-screen bypass LoRA (model 2750357).
- Hash verification for adapter files (full sha256, they are small).
- UI status via the `progress_cb` argument (drives shared.state.textinfo so
  Gradio shows the same progress the console does).

Gate handling: if a gated/private repo returns 401/404/GatedRepoError, we
print the exact accept-and-login steps and abort cleanly - never a loop.
"""

from .downloader import (  # noqa: F401
    AdapterMissing,
    civitai_download,
    download_adapter,
    ensure_adapters,
    hf_download,
    list_variants,
    verify_sha256,
)
