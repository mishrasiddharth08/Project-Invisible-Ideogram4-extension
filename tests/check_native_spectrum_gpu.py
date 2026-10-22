# requires: gpu
"""Explicit full-weight A/B check, with a 4 GiB process allocation cap.

Run only deliberately. Output artifacts are kept outside the extension.
This checks integration/output, not matched warm-cache performance.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

EXT = Path(__file__).resolve().parents[1]
FORGE = EXT.parent.parent
sys.path[:0] = [str(EXT), str(FORGE), str(FORGE / 'modules_forge/packages')]

import numpy as np
import psutil
import torch
from pi_ideogram_lib.pipeline import Ideogram4Pipeline
from pi_ideogram_lib.native_spectrum import DEFAULTS, _installed_node
from pi_ideogram_lib.sampler_configs import PRESETS
from pi_ideogram_lib import json_prompt
from pi_ideogram_forge.lora import find_uncond_replacement_lora
from pi_ideogram_forge.generate import _looks_blocked
from pi_ideogram_lora.runtime_lora import RuntimeLoraAdapter
from pi_ideogram_lora.ideogram4_lora import load_lora_state_dict


def main(parent):
    free, total = torch.cuda.mem_get_info()
    if free < 6 * 1024**3 or psutil.virtual_memory().available < 30 * 1024**3:
        raise RuntimeError('Need 6 GiB currently free VRAM and 30 GiB available RAM for this bounded check')
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / total)
    _installed_node()  # identical Forge attention/backend initialization for both runs
    output = Path(tempfile.mkdtemp(prefix='native-spectrum-validation-', dir=parent))
    models = FORGE / 'models'
    pipe = Ideogram4Pipeline(
        cond_path=models / 'Stable-diffusion/IDEOGRAM 4/Ideogram4-int8_convrot.safetensors',
        uncond_path=None,
        te_path=models / 'text_encoder/Qwen3-VL-8B-TextEncoder-int8_convrot--For-Ideogram4.safetensors',
        vae_path=models / 'VAE/FLUX.2-VAE-FP32--For-FLUX.2-Dev+Klein-4B-9B+ERNIE-Image+Ideogram4.safetensors')
    pipe._i4_intentional_single = True
    pipe.memory._test_vram_budget_gb = 4
    rows = []
    try:
        loaded = time.monotonic()
        pipe.load()
        load_seconds = time.monotonic() - loaded
        path = find_uncond_replacement_lora([models])
        if path is None:
            raise RuntimeError('Compatible local unconditional LoRA required')
        adapter = RuntimeLoraAdapter(pipe.conditional_transformer, load_lora_state_dict(path), 1.)
        assert adapter.install() > 0
        pipe.uncond_adapter = adapter
        preset = PRESETS['V4_TURBO_12']
        prompt, _, _ = json_prompt.build_prompt(mode='freeform-wrap',
            user_text='A red ceramic teapot on a wooden kitchen table beside a blue mug and oranges. Warm window light, detailed still-life photograph.',
            target_size=(1024, 1024))
        for label, options in [('off', None), ('forge-native', dict(DEFAULTS))]:
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            result = pipe(prompt, width=1024, height=1024, num_steps=12,
                guidance_schedule=preset.guidance_schedule, mu=preset.mu, std=preset.std,
                seed=1234, raise_on_caption_issues=False, native_spectrum=options,
                step_callback=lambda i, n: print(f'{label}: {i+1}/{n}', flush=True))
            torch.cuda.synchronize()
            pixels = np.asarray(result[0])
            result[0].save(output / f'{label}.png')
            if pixels.std() < 5:
                raise AssertionError('Nearly uniform output; inspect saved image')
            stats = pipe._native_spectrum_stats
            if options is not None and not stats['forecast_calls'] > 0:
                raise AssertionError('Native Spectrum did not forecast any branch calls')
            row = dict(mode=label, seconds=round(time.monotonic()-started, 2),
                peak_allocated_gib=round(torch.cuda.max_memory_allocated()/1024**3, 3),
                peak_reserved_gib=round(torch.cuda.max_memory_reserved()/1024**3, 3),
                native_stats=stats, blocked_output=bool(_looks_blocked(result[0])))
            rows.append(row)
            (output / 'measurements.json').write_text(json.dumps(dict(
                load_seconds=load_seconds, width=1024, height=1024,
                options=DEFAULTS, runs=rows), indent=2))
            print(json.dumps(row), flush=True)
        if any(row['blocked_output'] for row in rows):
            raise AssertionError(f'Blocked image detected; failure evidence retained at {output}')
        print(f'RESULTS: {output}', flush=True)
    finally:
        pipe.unload()


if __name__ == '__main__':
    main(sys.argv[1])
