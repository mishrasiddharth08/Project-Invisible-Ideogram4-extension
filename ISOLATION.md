# Project Invisible boundary

This release is prepared outside the running Forge tree. Applying it changes only the named Ideogram extension folder. The optional GitHub export is a separate clean source copy.

## What is deliberately not changed

- Forge source files, launchers, dependencies and virtual environment.
- Global CUDA/cuDNN settings.
- Forge's preset enum membership, private member maps and stock recipe tables.
- Another extension's Config-Presets files.
- Arbitrary loaded modules that happen to hold a process_images alias.
- SDXL, Flux, Qwen, video or other model math.
- Global CSS or other engines' controls.

## Runtime integration

A drop-in extension still needs host integration: the normal Generate/API entry points and Ideogram checkpoint registration use in-memory wrappers. These wrappers retain the original behavior for non-Ideogram calls. Optional attention/quantization delegates to Forge's implementations rather than vendoring host kernels.

V3 restores the native `ideogram4` UI Preset using an additive wrapper of the public `PresetArch.choices()` method and 33 Ideogram-owned option keys copied from Forge's real component templates. Enum members and every built-in recipe remain unchanged; later sibling preset registrations still work. The running process must be restarted after an update.

Signature checks disable incompatible integration. Unknown quantization descriptors fail with guidance. No automatic repair edits core files. This is fail-closed compatibility, not a promise that every future upstream update will be compatible.

## Removal / official updates

Close Forge fully before updating or removing the extension. Removing its folder and restarting removes its runtime code, styles and model hooks. Existing user model files, outputs and any old saved preference keys are user data and are not deleted automatically.

After a Forge update: restart, check for an Ideogram compatibility warning, then test one small image. If it fails, disable this extension and restart; use a compatible extension release. Do not “fix” it by modifying Forge core files.

Automated checks cover non-Ideogram passthrough, unrelated extension alias preservation, changed host signatures, shared-preset non-mutation, scoped UI and attention fallback. They cannot simulate every third-party extension or future Forge release.
