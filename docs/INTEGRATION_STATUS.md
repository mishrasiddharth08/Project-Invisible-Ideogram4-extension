# Blueprint application and acceptance — 22 October 2026

## Result

The extension now adds no model accordion, private generation tab or Generate
button. Model-specific choices moved to Forge Settings/Quicksettings using named
`pi_i4_*` options. GUI and API `override_settings` use the same request snapshot;
requests never rewrite saved settings. Legacy configuration remains readable.

The installed Forge commit is `10bb5ed977b3a551bacd999c29e8e370de7fc759`.
Its engine registry has no Ideogram engine or supported external registration
contract. This release therefore remains a **version-bound compatibility adapter**,
not full native-pipeline conformance. Achieving the blueprint's strict shared
processing, script/postprocess, resource reservation and sampler ownership gates
requires upstream host support. No Forge source or shared dependency was edited.

## Control map

| Intent | Current surface / behavior |
|---|---|
| Prompt, seed, dimensions, batch | Standard txt2img / img2img controls |
| Existing-image editing | Native img2img upload, resize, strength; experimental |
| Recipes / adapter choices | Settings → Ideogram 4 compatibility |
| Model source links / policies | Settings → Ideogram 4 setup & advanced |
| Programmatic settings | Named `pi_i4_*` keys in API `override_settings` |
| Euler integration | Standard Euler selection; actual logit-normal recipe recorded |
| Spectrum | Installed Forge SpectrumNode.patch through genuine UnetPatcher hooks |
| Errors | Propagate to Forge; no synthetic error image reported as success |
| Previews | Native image assignment, labeled initial gradient, real sampling frames, final decoded output |
| Progress | Current-image and overall counters; encoding/sampling/decoding/saving phases |

The private Spectrum implementation was removed and archived. Forge's existing
Spectrum Integrated GUI/API arguments now drive its installed SpectrumNode.patch
through genuine UnetPatcher hooks. Flow velocities map to denoised endpoints for
the native forecaster; actual forward results are retained exactly. CFG branches
have separate histories, and adapter changes, skipped branches and cancellation
invalidate/release owned state. No built-in Spectrum source was copied or edited.
See [NATIVE_SPECTRUM.md](NATIVE_SPECTRUM.md) for current evidence and usage.

Negative text conditioning, variation seeds, prompt/seed arrays, Hires fix,
inpainting, outpainting and reference-conditioned instruction editing are not
advertised. Unsupported requests fail explicitly rather than being ignored.
Dimensions must be 256–2048 and divisible by 16. Sequential image execution is a
memory policy, not a claim of tensor batching. Existing logs/infotext expose the
actual Ideogram Euler/schedule route, not an arbitrary selected host scheduler.

## Verification on this revision

- Initial blueprint baseline: 48 Python tests plus 14 subtests. The subsequent
  native Spectrum revision passed 72 tests plus 14 subtests; see its report.
- Native Settings/UI contract check passed: no accordions, no positional UI
  parameters; named request overrides and legacy defaults preserved.
- JavaScript two-progress-bar check passed.
- All 24 console-progress regression checks passed.
- Real tiny-transformer sampling, adapter cleanup and cache comparisons passed
  on CPU. This is numerical/component evidence, not a full-weight GPU test.
- Switch-cleanup tests cover deferred release, rapid checkpoint changes and
  failure of one unload without abandoning other owned models.
- Preview tests cover native assignment, gradient/final behavior and progress.

Reproduction: run the focused `test_request_contract`, `test_blueprint_contract`,
`test_switch_release`, `test_final_preview`, `test_img2img_pipeline`,
`test_img2img_integration`, `test_gray_batch_and_projection`,
`test_refiner_contract`, and `test_release_integrity` suites with Forge's Python;
run `tests/test_ui_layout.py` and `node tests/test_dual_progress.cjs` separately.

Full-weight standalone GPU A/B images at 1024px were subsequently generated and
visually checked for the native Spectrum revision; see its report. Browser/API
end-to-end generation, physical low-VRAM/AMD tests, OOM/cancel soak and full script
ecosystem compatibility remain **Not Tested**. A 4 GiB software allocation cap on
an RTX 5090 is not physical low-VRAM hardware validation. This still does not
satisfy all blueprint release gates or create a full native Ideogram engine.

## Cleanup and recovery

Unrelated work trees, duplicate release checkout, old backups, quote utilities,
generated images, the global attention patch and private Spectrum source were
moved into `G:\Project-Invisible-Backups\ideogram4-cleanup-20261022`.
They remain recoverable. Ideogram code, settings, optional correction asset,
licenses, documentation, version history and relevant tests remain in the extension.

Three old image-evidence directories under `tests` remain inaccessible:
`img2img-check-ubjxdbxm`, `memory-preview-9ajcodep`, and `memory-preview-fs4awlvx`.
Windows denied ordinary moves; automatic approval review rejected taking ownership
and changing their permissions. They were left untouched. Permission changes
require the user's explicit approval before cleanup can finish.

Restart Forge to unload the old in-memory UI/hooks and activate this revision.
No changes from this task were pushed to GitHub.
