# V3 audit fixes — 2026-09-10

- Restore the native top-left `ideogram4` UI Preset removed in V2, without enum-member surgery or Forge core edits; preserve existing settings and stock presets.
- Include text encoder, VAE, checkpoint/adapter file identity and adapter policy in the pipeline cache. Keep unchanged selections fast; reload changed weights.
- Prevent permanently merged TurboTime pipelines leaking into other recipes.
- Release runtime adapter owners and streaming block hooks when unloading or dropping guidance weights.
- Keep multiple runtime LoRAs active together; honor signed and zero strengths, and UI argument slices starting at zero.
- Preserve weights when Forge converts header quantization metadata in place; remap encoder markers together with their original weights.
- Route rotated FP8 through Forge and accept scalar FP8 scales with singleton dimensions.
- Add 26 regression tests plus a real installed-Forge preset/Gradio integration check.

# V2 release candidate — 2026-09-10

- Compact exact text-feature capture and two-entry LRU; unload releases prompt tensors.
- Skip unconditional zero-feature projection; avoid redundant CPU offload cleanup.
- RAM-backed block streaming with runtime adapter device alignment.
- Preserve checkpoint precision and sampling recipe; disable silent quality-reducing OOM recovery.
- Recognize modern quantization descriptors and reject unsupported/missing backends.
- Text-encoder NF4/mixed-dense loading; preserve all converted quantization scales and reject missing language weights.
- Delegate compatible attention to Forge; preserve masked attention semantics.
- Retain Spectrum; add opt-in experimental First Block Cache with separate histories and no stacking.
- Simplify recipes, acceleration and adapter controls; scope styling.
- Move Forge helpers into pi_ideogram_forge; retain only engine.py in scripts.
- Namespace configuration, prompt and asset imports so generic module names cannot collide with other extensions.
- Stop mutating shared Forge preset enums, global cuDNN flags and Config-Presets files.
- Remove the remaining hidden debanner download; keep local optional assets intact.
- Narrow API alias interception to known Forge modules; add host signature guards.
- Apache-2.0 license, upstream notices, regression checks and measured limitations.
