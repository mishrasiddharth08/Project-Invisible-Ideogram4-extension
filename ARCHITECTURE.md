# Architecture

## Entry points and ownership

- `scripts/engine.py`: sole Forge script, scoped Gradio panel, ten returned controls.
- `pi_ideogram_forge/run.py`: selection-gated Generate/API dispatch and pipeline lifetime.
- `pi_ideogram_forge/generate.py`: resolved settings → one generation.
- `pi_ideogram_forge/{paths,selection,dropdown,routing,picker,lora}.py`: integration helpers in an extension-specific namespace.
- `pi_ideogram_lib`: Ideogram model, text encoder, VAE, sampler, format loading and memory.
- `pi_ideogram_lora`: adapter math and runtime side paths.
- `pi_ideogram_lib/config.py`: extension-local configuration. Never include the generated root config.json in Git.
- `pi_ideogram_lib/json_prompt.py`: isolated prompt formatting and token budgeting.
- `pi_ideogram_assets`: namespaced model links and explicit asset utilities.
- `pi_ideogram_forge/legacy_preset.py`: native top-left `ideogram4` preset through an additive public choices wrapper and engine-owned options; no enum-member or stock-recipe mutation.

UI slot order is defined once in `pi_ideogram_lib/settings.py::UI_KEYS`. Slot 9 retains its legacy name, spectrum_enabled, for backward compatibility; its dropdown accepts off/spectrum/first_block, and old Boolean inputs still map to off/Spectrum. Settings resolve this into mutually exclusive Boolean flags. No second approximation is enabled implicitly.

## Quality and memory invariants

Prompt encoding runs the original full Qwen sequence. Only captured TEXT prefixes are retained between taps. The two-entry LRU stores compact native-precision features; padded image slots are reconstructed for the transformer contract. Unload releases cache tensors.

An explicit llm_features=None indicates an image-only guidance pass. Its redundant zero-feature normalization/projection is omitted. Conditioned passes retain their original computation.

Block streaming wraps only this pipeline's modules. Weights visit CUDA just before their block and return to RAM afterward, including exceptions. Nothing is re-quantized. Rotary initialization uses CUDA numerics before parking its tiny buffer. Runtime adapter factors follow the activation device.

Spectrum owns separate velocity histories. First Block Cache owns per-model/per-guidance histories through a ContextVar whose lifetime is one pipeline call. Both reset after cancellation or failure. First Block Cache allows at most one consecutive reused evaluation, four full warmup steps, and two full tail steps. Neither enables the other.

## Forge boundary

Non-Ideogram calls forward the original positional/keyword arguments and return values. The API bridge only touches known Forge entry-point bindings, not arbitrary other extensions.

The attention bridge reads the host's already-loaded backend module without booting Forge. It checks the function contract and delegates only compatible unmasked calls. Masked calls use SDPA. It never replaces the global PyTorch attention function.

Modern quantization uses Forge's own operations context, with construction serialized by the shared extension lock. All extension model builders use that lock. Forge owns the temporary operations replacement and restoration. This is still an upstream API dependency, not a guarantee against arbitrary concurrent third-party monkey-patching.

No process-global backend settings are changed. Full restart is required after installation, removal or updates.

## Update behavior

Changed required host signatures disable the integration rather than inventing a replacement. Missing quantization capabilities stop the affected load. Stock model listing completes even if Ideogram-specific registration fails.

Do not edit Forge's modules, backend, extensions-builtin, launch scripts or dependency environment. Do not add hidden downloads or persist credentials in a release. Review the isolated updater's manifest and hashes when distributing an update.
