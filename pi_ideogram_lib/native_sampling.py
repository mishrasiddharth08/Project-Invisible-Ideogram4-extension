"""Bridge Forge's installed sigma samplers/schedulers to Ideogram VE sampling."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
import torch.nn.functional as F

_SPECIAL = {"ddim", "plms"}
_VERIFIED_FUNCTIONS = {
    "sample_dpmpp_2m", "sample_euler", "sample_euler_ancestral", "sample_lms",
    "sample_heun", "sample_dpm_2", "sample_res_multistep", "sample_Kohaku_LoNyu_Yog",
    "restart_sampler", "sample_gradient_estimation", "sample_res_multistep_ancestral",
    "sample_euler_a2", "sample_res_solver", "sample_clyb_4m_sde_momentumized",
    "sample_ipndm", "sample_ipndm_v", "sample_deis", "sample_ddpm", "sample_heunpp2",
    "sample_dpm_2_ancestral", "sample_dpm_fast", "sample_dpmpp_2m_sharp",
    "sample_res_2s_nc", "sample_res_2m_nc", "sample_res_2s_nc_sharp", "sample_res_2m_nc_sharp",
    "sample_dpmpp_sde", "sample_dpmpp_2m_sde", "sample_dpmpp_3m_sde", "sample_lcm",
    "sample_er_sde",
}
_RESERVED = {"model", "x", "sigmas", "sigma_sched", "n", "sigma_min",
             "sigma_max", "extra_args", "callback", "disable"}


def _named(items, value, fields):
    key = str(value or "").strip().casefold()
    found = []
    for item in items:
        names = []
        for field in fields:
            names.extend(getattr(item, field, None) or [] if field == "aliases"
                         else [getattr(item, field, None)])
        if key in {str(x).casefold() for x in names if x is not None}:
            found.append(item)
    if len(found) != 1:
        raise ValueError(f"Unknown or ambiguous {fields[0]} '{value}'.")
    return found[0]


def _sampler_function(entry):
    import k_diffusion.sampling as sampling

    constructor = entry.constructor
    values = list(getattr(constructor, "__defaults__", None) or ())
    values += list((getattr(constructor, "__kwdefaults__", None) or {}).values())
    values += [c.cell_contents for c in (getattr(constructor, "__closure__", None) or ())]
    candidates = []
    for value in values:
        func = getattr(sampling, value, None) if isinstance(value, str) else value
        if callable(func):
            try:
                params = inspect.signature(func).parameters
            except (TypeError, ValueError):
                continue
            if any(x in params for x in ("sigmas", "sigma_sched", "n")):
                candidates.append((func, value if isinstance(value, str) else func.__name__))
    unique = {id(func): (func, name) for func, name in candidates}
    if len(unique) != 1:
        raise ValueError(f"Sampler '{entry.name}' is not a supported conventional sigma sampler.")
    func, name = next(iter(unique.values()))
    identity = f"{entry.name} {name} {getattr(func, '__module__', '')}".casefold()
    unsupported = ("cfg++", "cfg_pp", "cfgpp", "unipc", "res4lyf")
    if any(token in identity for token in unsupported):
        raise ValueError(f"Sampler '{entry.name}' requires an unsupported Forge model/CFG contract.")
    if name not in _VERIFIED_FUNCTIONS:
        raise ValueError(f"Sampler '{entry.name}' is installed but not verified for the Ideogram VE wrapper.")
    return func, name


def resolve_request(p) -> dict:
    """Resolve exact installed Forge entries; never default or substitute."""
    from modules import sd_samplers, sd_samplers_kdiffusion, sd_schedulers, shared

    requested = str(getattr(p, "sampler_name", "") or "").strip()
    if requested.casefold() in _SPECIAL:
        raise ValueError(f"Sampler '{requested}' uses Forge's unsupported special timestep contract.")
    entry = _named(sd_samplers.all_samplers, requested, ("name", "aliases"))
    if entry.name.casefold() in _SPECIAL:
        raise ValueError(f"Sampler '{entry.name}' uses Forge's unsupported special timestep contract.")
    if any(token in entry.name.casefold() for token in ("cfg++", "unipc", "res4lyf")):
        raise ValueError(f"Sampler '{entry.name}' requires an unsupported Forge model/CFG contract.")
    func, function_name = _sampler_function(entry)

    scheduler_value = str(getattr(p, "scheduler", "Automatic") or "Automatic")
    scheduler = _named(sd_schedulers.schedulers, scheduler_value, ("label", "aliases", "name"))
    if entry.name == 'DPM++ 2M SDE Heun' and scheduler.name == 'exponential':
        raise ValueError('DPM++ 2M SDE Heun / Exponential failed real Ideogram image validation. '
                         'Choose Automatic, or iPNDM / SGM Uniform. No sampler was substituted.')
    if scheduler.name in {"align_your_steps", "turbo"}:
        raise ValueError(f"Scheduler '{scheduler.label}' requires unavailable Forge model attributes.")
    function_params = inspect.signature(func).parameters
    consumes_schedule = "sigmas" in function_params or "sigma_sched" in function_params
    if scheduler.label != "Automatic" and not consumes_schedule:
        raise ValueError(f"Sampler '{entry.name}' owns its internal schedule and cannot honor "
                         f"scheduler '{scheduler.label}'. Use Automatic.")
    sampler_kwargs = {key: value for key, value in dict(entry.options).items()
                      if key in function_params and key not in _RESERVED}
    extra_names = sd_samplers_kdiffusion.sampler_extra_params.get(func,
                   sd_samplers_kdiffusion.sampler_extra_params.get(function_name, []))
    for name in set(extra_names) | {"eta"}:
        if name not in inspect.signature(func).parameters:
            continue
        value = getattr(p, name, None)
        if value is None:
            value = getattr(shared.opts, name, None)
        if value is not None:
            sampler_kwargs[name] = value
    supplied = getattr(p, "native_sampler_options", None) or {}
    if not isinstance(supplied, dict):
        raise ValueError("native_sampler_options must be a dictionary.")
    valid = set(inspect.signature(func).parameters) - _RESERVED
    unknown = set(supplied) - valid
    if unknown:
        raise ValueError(f"Unsupported native sampler options: {sorted(unknown)}")
    sampler_kwargs.update(supplied)
    extra_args = getattr(p, "native_sampler_extra_args", None) or {}
    if not isinstance(extra_args, dict):
        raise ValueError("native_sampler_extra_args must be a dictionary.")

    scheduler_kwargs = getattr(p, "native_scheduler_options", None) or {}
    if not isinstance(scheduler_kwargs, dict):
        raise ValueError("native_scheduler_options must be a dictionary.")
    schedule_contract = "explicit_sigmas" if consumes_schedule else "solver_owned_endpoints"
    return {
        "function": func,
        "sampler_name": entry.name,
        "scheduler_name": scheduler.label,
        "options": {**dict(entry.options), "schedule_contract": schedule_contract,
                    "sampler_kwargs": sampler_kwargs,
                    "scheduler_kwargs": dict(scheduler_kwargs), "extra_args": dict(extra_args)},
        "function_name": function_name,
    }


class _SigmaProxy:
    def __init__(self, official, shift=1.0):
        positive = official[:-1].detach().float().cpu()
        if len(positive) == 1:
            dense = positive.repeat(1000)
        else:
            dense = F.interpolate(positive.log()[None, None], size=1000,
                                  mode="linear", align_corners=True)[0, 0].exp()
            dense[0], dense[-1] = positive[0], positive[-1]
        self.sigmas = dense.flip(0)
        self.log_sigmas = self.sigmas.log()
        predictor = SimpleNamespace(shift=float(shift))
        unet = SimpleNamespace(model=SimpleNamespace(predictor=predictor))
        self.inner_model = SimpleNamespace(forge_objects=SimpleNamespace(unet=unet))

    def sigma_to_t(self, sigma, quantize=None):
        q = torch.as_tensor(sigma, dtype=self.log_sigmas.dtype).log().cpu()
        if self.log_sigmas[0] == self.log_sigmas[-1]:
            return torch.zeros_like(q)
        hi = torch.searchsorted(self.log_sigmas, q).clamp(1, len(self.log_sigmas) - 1)
        lo = hi - 1
        frac = (q - self.log_sigmas[lo]) / (self.log_sigmas[hi] - self.log_sigmas[lo])
        return (lo + frac).to(q.device)

    def t_to_sigma(self, t):
        t = torch.as_tensor(t, dtype=self.log_sigmas.dtype).cpu().clamp(0, len(self.log_sigmas) - 1)
        lo = t.floor().long()
        hi = t.ceil().long()
        frac = t - lo
        return torch.exp(self.log_sigmas[lo] + frac * (self.log_sigmas[hi] - self.log_sigmas[lo]))

    def get_sigmas(self, n=None):
        if n is None:
            return torch.cat((self.sigmas.flip(0), self.sigmas.new_zeros(1)))
        t = torch.linspace(len(self.sigmas) - 1, 0, n)
        return torch.cat((self.t_to_sigma(t), self.sigmas.new_zeros(1)))


def _validate_sigmas(value, expected, device, label):
    sigmas = torch.as_tensor(value, dtype=torch.float32, device=device)
    if sigmas.ndim != 1 or len(sigmas) < 2 or (expected is not None and len(sigmas) != expected):
        raise ValueError(f"{label} returned shape {tuple(sigmas.shape)}; expected one-dimensional {expected} sigmas.")
    if not torch.isfinite(sigmas).all() or sigmas[-1] != 0 or not (sigmas[:-1] > 0).all():
        raise ValueError(f"{label} returned a non-finite, non-positive, or unterminated schedule.")
    if not (sigmas[:-1] > sigmas[1:]).all():
        raise ValueError(f"{label} returned a non-descending schedule.")
    return sigmas


def build_sigmas(request, official_sigmas, width, height, device):
    """Build a strict descending VE schedule bounded by official endpoints."""
    official = _validate_sigmas(official_sigmas, len(official_sigmas), device, "Official Ideogram")
    if request["scheduler_name"] == "Automatic":
        request["options"]["schedule_count"] = len(official) - 1
        return official

    from modules import sd_schedulers, shared
    scheduler = _named(sd_schedulers.schedulers, request["scheduler_name"], ("label", "aliases", "name"))
    if scheduler.function is None:
        raise ValueError(f"Scheduler '{scheduler.label}' has no native function.")
    opts = dict(request["options"].get("scheduler_kwargs", {}))
    allowed = set(inspect.signature(scheduler.function).parameters)
    unknown = set(opts) - (allowed | {"shift"})
    if unknown:
        raise ValueError(f"Unsupported native scheduler options: {sorted(unknown)}")
    proxy = _SigmaProxy(official, opts.pop("shift", 1.0))
    steps, sigma_min, sigma_max = len(official) - 1, official[-2].item(), official[0].item()
    kwargs = {"n": steps, "sigma_min": sigma_min, "sigma_max": sigma_max, "device": device}
    if "inner_model" in allowed:
        kwargs["inner_model"] = proxy
    if "width" in allowed:
        kwargs["width"] = width
    if "height" in allowed:
        kwargs["height"] = height
    if "rho" in allowed:
        native_rho = getattr(shared.opts, "rho", 0)
        kwargs["rho"] = opts.get("rho", native_rho or scheduler.default_rho)
    kwargs.update(opts)
    kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    missing = [n for n, p in inspect.signature(scheduler.function).parameters.items()
               if p.default is inspect.Parameter.empty and n not in kwargs]
    if missing:
        raise ValueError(f"Scheduler '{scheduler.label}' requires unsupported arguments: {missing}")
    raw = torch.as_tensor(scheduler.function(**kwargs), dtype=torch.float32, device=device).flatten()
    if scheduler.name == "kl_optimal" and len(raw) and raw[-1] > 0:
        raw = torch.cat((raw, raw.new_zeros(1)))
        request["options"]["schedule_adjustment"] = "KL Optimal native positive endpoint: appended terminal zero"
    if scheduler.name in {"flow_match", "flux2"}:
        raw = _validate_sigmas(raw, None, device, scheduler.label + " bounded")
        if raw[0] > 1.000001:
            raise ValueError(f"Scheduler '{scheduler.label}' did not return a bounded [1,0] schedule.")
        bounded = raw[:-1].clamp(0, 1)
        positive = (bounded / (1 - bounded).clamp_min(torch.finfo(bounded.dtype).eps)).clamp(
            min=official[-2], max=official[0])
        positive[0] = official[0]
        raw = torch.cat((positive, positive.new_zeros(1)))
    result = _validate_sigmas(raw, None, device, scheduler.label)
    if result[0] > official[0] * 1.00001 or result[-2] < official[-2] * 0.99999:
        raise ValueError(f"Scheduler '{scheduler.label}' exceeded official Ideogram VE endpoints.")
    request["options"]["schedule_count"] = len(result) - 1
    request["options"]["requested_schedule_count"] = len(official) - 1
    return result


def run_sampler(request, denoiser, x_ve, sigmas, callback, seed):
    """Invoke the resolved native callable while preserving caller RNG state."""
    from backend.modules.k_prediction import AbstractPrediction

    class _VEDenoiser:
        def __init__(self, function):
            self.function = function
            self.inner_model = SimpleNamespace(
                predictor=AbstractPrediction(sigma_data=1.0, prediction_type="epsilon"))

        def __call__(self, *args, **kwargs):
            return self.function(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.function, name)

    func = request["function"]
    params = inspect.signature(func).parameters
    kwargs = dict(request["options"].get("sampler_kwargs", {}))
    if "sigmas" in params:
        kwargs["sigmas"] = sigmas
    if "sigma_sched" in params:
        kwargs["sigma_sched"] = sigmas
    if "n" in params:
        kwargs["n"] = len(sigmas) - 1
    if "sigma_min" in params:
        kwargs["sigma_min"] = sigmas[-2]
    if "sigma_max" in params:
        kwargs["sigma_max"] = sigmas[0]
    seed = 0 if seed is None else int(seed)
    if "extra_args" in params:
        kwargs["extra_args"] = {"seed": seed, **request["options"].get("extra_args", {})}
    if "callback" in params:
        kwargs["callback"] = callback
    if "disable" in params:
        kwargs["disable"] = True
    if not any(x in params for x in ("sigmas", "sigma_sched", "n")):
        raise ValueError(f"Sampler '{request['sampler_name']}' has no supported sigma contract.")
    # manual_seed touches every CUDA RNG; preserve all of them, not just the
    # generation device, so another GPU's unrelated workload is unaffected.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        result = func(_VEDenoiser(denoiser), x_ve, **kwargs)
    if not isinstance(result, torch.Tensor) or result.shape != x_ve.shape or not torch.isfinite(result).all():
        raise ValueError(f"Sampler '{request['sampler_name']}' returned an invalid tensor.")
    return result
