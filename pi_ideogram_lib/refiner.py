"""Transient refiner policy; ordinary Ideogram generation keeps its settings."""
import math


def neutral_options(options):
    result = dict(options)
    result.update(preset='Auto', turbo=False, adapter_choice='',
                  prompt_upsampling=False, filter_bypass=False, bypass_lora=False,
                  bypass_lora_name='', bypass_strength=0.0,
                  sigma_smooth=False, sigma_smooth_steps=0, layer_dampen=False,
                  debanner_enabled=False, spectrum_enabled=False,
                  first_block_cache_enabled=False, use_uncond_lora=False,
                  allow_quality_reducing_oom_recovery=False)
    return result


def fixed_step_counts(steps, strength):
    steps = max(1, int(steps))
    strength = float(strength)
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('img2img strength must be between 0 and 1')
    if strength == 0:
        return steps, 0
    schedule = steps / strength
    if not math.isfinite(schedule):
        raise ValueError('img2img strength is too small for a finite schedule')
    return max(steps, math.ceil(schedule)), steps
