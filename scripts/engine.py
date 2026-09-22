"""PROJECT INVISIBLE - Ideogram 4 for Forge Neo (branch: neo).

STRICT INVISIBILITY SPEC (user mandate):
- Ideogram 4 checkpoints appear in the NORMAL dropdown, labelled
  "Ideogram4 - <name>" (registration owned by pi_ideogram_forge/dropdown.py).
- The stock Generate button runs the official dual-model pipeline when (and
  only when) an Ideogram 4 checkpoint is selected (pi_ideogram_forge/run.py owns the
  takeover). Everything else is 100% stock.
- Minimal visible UI: an "Ideogram 4 Controls" accordion appears only when an
  Ideogram 4 checkpoint is selected. It contains the PRESET / QUANT / UNCOND
  selectors and the FILTER BYPASS panel (checkbox + strength master + advanced
  levers). All other controls are the stock Forge ones.
- Delete this folder -> Forge is stock again. Nothing else anywhere.

THIS FILE is the Forge-facing ADAPTER only: the boot sys.path wiring, the
Script subclass (accordion UI + visibility), and the two installs (takeover
wrapper + dropdown hook).  Every concern below lives in ONE sibling module
(see ARCHITECTURE.md):
    paths.py      - where the model dirs are in this Forge install
    selection.py  - which checkpoint is selected / is it Ideogram 4
    dropdown.py   - "Ideogram4 - <name>" registration in the stock dropdown
    run.py        - the takeover (_wrapped_run), session state, pipeline cache,
                    generate() facade, blocked-image auto-escalation
    generate.py   - sampling orchestration over a RESOLVED settings dict
    picker.py     - Model Setup (spec F) markdown views
    routing.py    - detected-format policy surface (render + run-time enforce)
The names those modules own are re-exported here so harnesses/tests keep one
Forge-facing surface; test seams patch the OWNER modules, not this one.

DATA FLOW (single direction): always-visible process() only runs inside
processing.process_images, which this takeover skips - so the accordion
values are sliced straight out of p.script_args in run._wrapped_run via
_resolve_run_opts (args_from/args_to on our Script instance) and turned into
RESOLVED run settings by pi_ideogram_lib.settings. pi_ideogram_forge/generate.py only
ever sees that resolved dict; it never reads Gradio or config.json.

img2img: the official model is t2i-only; a clear text card explains it.

Assets are NOT downloaded by this extension. The Model Setup panel prints
clickable links and the folder each file belongs in; downloads happen in the
browser, where the user is already signed in. There is no API key anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXT_DIR = Path(__file__).resolve().parent.parent
if str(_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXT_DIR))

# standalone/unit-test support: Forge vendors gguf under modules_forge/packages
_FORGE_ROOT = _EXT_DIR.parent.parent
if str(_FORGE_ROOT) not in sys.path:
    sys.path.append(str(_FORGE_ROOT))
for _vendored in ("modules_forge/packages",):
    _vp = _FORGE_ROOT / _vendored
    if _vp.is_dir() and str(_vp) not in sys.path:
        sys.path.append(str(_vp))

import pi_ideogram_lib.settings as pi_settings  # noqa: E402  (single owner of run settings)

# Forge-interface concerns, one module each (see ARCHITECTURE.md).  These
# imports ARE the re-export surface: engine binds the names so the Script UI
# and external harnesses reach the single owner for each behavior.
from pi_ideogram_forge.paths import models_dirs as _models_dirs  # noqa: E402
from pi_ideogram_forge.selection import (  # noqa: E402
    _ci_is_ideogram4,
    _current_ideogram_file,
    _resolve_ci_for_name,
    _selected_checkpoint_name,
    _selected_ci_is_i4,
    is_ideogram4_selected,
)
from pi_ideogram_forge.dropdown import install_dropdown_hook  # noqa: E402
from pi_ideogram_forge.routing import (  # noqa: E402
    _enforce_loadable_policy,
    _policy_for_current,
    _routing_markdown,
)
from pi_ideogram_forge.picker import (  # noqa: E402
    _on_variant_changed,
    _picker_cfg,
    _picker_markdown,
    _picker_quant,
    _variant_card_markdown,
)
from pi_ideogram_forge.run import (  # noqa: E402  (takeover + session state)
    _ACTIVE_LORAS,
    _maybe_download_adapters,
    _next_bypass_anchor,
    _ORIGINAL_RUN,
    _PIPELINE_CACHE,
    _resolve_run_opts,
    _wrapped_run,
    generate,
    install_wrapper,
)
# generate.py helpers re-exported for harnesses/tests (the 1024-GPU suites).
from pi_ideogram_forge.generate import (  # noqa: E402
    generate as _generate_fn,
    _looks_blocked,
    _resolve_sampling,
    _resolve_seed,
    _safe_comment,
)
import pi_ideogram_forge.lora as lora  # noqa: E402  (boot: lora.set_models_dirs below)

# Register 'ideogram4' in Forge Neo's native top-left "UI Preset" dropdown.
try:
    from pi_ideogram_forge import legacy_preset as forge_preset  # noqa: E402
    forge_preset.install()
except Exception as _e:
    print(f"[Invisible-I4] WARNING: forge_preset unavailable: {_e}")

try:
    from modules import scripts
    _FORGE_OK = True
except Exception as _e:
    print(f"[Invisible-I4] WARNING: Forge imports failed: {_e}")
    scripts = None
    _FORGE_OK = False

TAG = "[Invisible-I4]"


# --------------------------------------------------------------------------- #
# hardware autodetection -> config.json (first launch / GPU or driver change)
# --------------------------------------------------------------------------- #

def _quickstart() -> str:
    """Six lines a first-time user needs, and nothing else.

    Everything here is a question someone actually hit: which preset is the
    fast one (V4_TURBO_12 is NOT it), how to attach a LoRA, why steps and CFG
    are not in this panel, and why bigger canvases cost so much.
    """
    return (
        "Write a prompt → choose a recipe → **Generate**. Start with **Balanced**; memory is managed automatically."
    )


def _recipe_line(preset_name: str) -> str:
    """One line stating exactly what the selected preset will run.

    The preset dropdown used to be an opaque code name: picking V4_TURBO_12
    told you nothing about steps, CFG or shift, so the only way to learn what
    a preset did was to generate and read the console. Since these are the
    OFFICIAL numbers and the whole point is that they are not tampered with,
    showing them is the cheapest way to make the panel trustworthy.
    """
    try:
        from pi_ideogram_lib.sampler_configs import PRESETS
        import pi_ideogram_lib.settings as _st
        name = str(preset_name or "")
        if name == _st.PRESET_AUTO_LABEL or name == "Auto":
            return ("**Auto** - follows Forge's own steps and CFG sliders. "
                    "Pick a named preset for the official recipe.")
        if name == "TurboTime-4":
            return ("**TurboTime-4** - 4 steps, CFG 1.0, shift mu 0.5 / std 1.75. "
                    "Needs the TurboTime distill LoRA.")
        if name == "Instant-8":
            return "**Instant-8** - 8 steps, CFG 1.0, shift mu 0.5 / std 1.75. *(official)*"
        pr = PRESETS.get(name)
        if pr is None:
            return ""
        sched = list(pr.guidance_schedule)
        # The schedule is stored in LOOP-INDEX order: index 0 is the FINAL
        # step. Read it back the way a person thinks about it - main pass
        # first, polish tail last.
        tail = sum(1 for g in sched if g == 3.0)
        if tail:
            _plural = "step" if tail == 1 else "steps"
            cfg = (f"CFG {max(sched):g} for {len(sched) - tail} steps, "
                   f"then {tail} polish {_plural} at 3.0")
        else:
            cfg = f"CFG {max(sched):g} throughout"
        # "V4_TURBO_12" reads as "the turbo one", so people pick it expecting
        # the TurboTime distill LoRA. It is not that: it is the official
        # 12-step CFG 7->3 recipe, it loads the 9.3 GB uncond half, and it runs
        # two forwards per step. Someone chose it, waited 12.4 s/it, and asked
        # whether Turbo was working. Say which presets actually use the LoRA.
        note = ("  *Official CFG recipe - does NOT use the TurboTime LoRA. "
                "For that, pick TurboTime-4 or Instant-8.*" if name == "V4_TURBO_12" else "")
        return (f"**{name}** - {pr.num_steps} steps, {cfg}, "
                f"shift mu {pr.mu} / std {pr.std}. *(official)*" + note)
    except Exception:
        return ""


def _bypass_help(has_any: bool) -> str:
    """What to do when the bypass dropdown has nothing in it.

    The list only offers adapters that are on disk AND classify as bypasses
    (the uncond-replacement and TurboTime LoRAs were being offered here, and
    applying either at -0.25 is not a bypass). That is correct, but it can
    leave the dropdown holding only "None", which reads as broken.
    """
    if has_any:
        return "Choose an installed bypass adapter to help reduce gray blocked images. Results depend on the prompt and checkpoint."
    return ("No bypass adapter installed. Get `Gray_000002000.safetensors` from **Model Setup**, "
            "place it in `models/Lora/IDEOGRAM LORA`, then choose **Refresh adapters**.")


def _bypass_status(name, strength, raw_prompt=False):
    """Report the user's configuration, not an unverified successful application."""
    selected = bool(name) and str(name) != pi_settings.BYPASS_NONE_LABEL
    amount = float(strength or 0.0)
    state = (f"Ready · first-step adapter at {amount:+.2f}" if selected and amount
             else "Off · select an adapter" if not selected else "Off · strength is zero")
    if selected and amount > 0:
        state += ". Positive strength reverses the usual Gray bypass direction."
    return state + (" · Prompt structure assistance is off (raw prompt)." if raw_prompt
                    else " · Prompt structure assistance is on.")


def _probe_hardware_once() -> None:
    """First-launch / GPU-change probe. Delegates to hardware.configure_once()
    (single owner of the probe->config.json policy, shared with install.py)."""
    from pi_ideogram_lib.hardware import configure_once
    res = configure_once()
    if not res or not res.get("changed"):
        return
    p, prof, pd = res["probe"], res["profile"], res["defaults"]
    for w in res.get("warnings", []):
        print(f"{TAG} WARNING: {w}")
    print(f"{TAG} hardware: profile={prof} vram={p.get('vram_total_gb')}GB "
          f"recommended quant={pd['quant_auto']} max_size={pd['max_size']} "
          f"uncond-LoRA={pd['use_uncond_lora_default']}")

try:
    _probe_hardware_once()
except Exception:
    pass

try:
    lora.set_models_dirs(_models_dirs())
except Exception:
    pass


# --------------------------------------------------------------------------- #
# the Script: AlwaysVisible, ZERO changes to stock pages.
# ui() is the accordion (13 RETURNED controls = settings.UI_KEYS contract) +
# the Model Setup picker section (created but NOT returned: Forge slices only
# returned controls into script.args_from:args_to - ARCHITECTURE rule 3).
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# accordion visibility. ui() registers its (accordion, detect-markdown) pair
# here; the hook below wires every registered pair to Forge's real checkpoint
# dropdown once that dropdown exists. One shared handler, so the txt2img and
# img2img panels stay in step.
# --------------------------------------------------------------------------- #

_I4_PANELS: list = []


def _i4_visibility_updates(ckpt_name, panels):
  """One (visible, markdown) update pair per registered panel."""
  import gradio as gr
  try:
    is_i4 = _selected_ci_is_i4(ckpt_name)
    ci = _resolve_ci_for_name(ckpt_name)
    fn = getattr(ci, "filename", None) if ci else None
    md = _routing_markdown(fn) if (is_i4 and fn) else "No Ideogram 4 checkpoint selected."
  except Exception:
    is_i4, md = False, "No Ideogram 4 checkpoint selected."
  updates = []
  for _ in panels:
    updates.extend((gr.update(visible=is_i4), gr.update(value=md)))
  return updates


def install_visibility_hook():
  """Wire the panels to the checkpoint dropdown when Forge creates it.

  after_component fires for every component as it is built, so this runs at the
  one moment the dropdown is both real and still inside the Blocks graph - the
  same approach the sibling PROJECT INVISIBLE extensions use.  Failures print;
  they are no longer swallowed.
  """
  try:
    from modules import script_callbacks
  except Exception as _e:
    print(f"{TAG} WARNING: visibility hook unavailable: {_e}")
    return

  def _after_component(component, **kwargs):
    elem = kwargs.get("elem_id", getattr(component, "elem_id", None))
    if elem != "setting_sd_model_checkpoint" or not _I4_PANELS:
      return
    try:
      from gradio.context import Context
      if Context.root_block is None:
        return  # transient component built outside the UI graph
      panels = list(_I4_PANELS)
      component.change(fn=lambda name: _i4_visibility_updates(name, panels),
                       inputs=[component],
                       outputs=[c for pair in panels for c in pair],
                       queue=False)
    except Exception as _e:
      print(f"{TAG} WARNING: could not wire panel visibility: {_e}")

  script_callbacks.on_after_component(_after_component)


if _FORGE_OK and scripts is not None:

  class Ideogram4EngineScript(scripts.Script):
    def title(self):
      return "Ideogram 4 (Invisible)"

    def show(self, is_img2img):
      return scripts.AlwaysVisible

    # Marker so the ScriptRunner wrapper (run._resolve_run_opts) can find THIS
    # script's instance and slice its accordion values out of p.script_args.
    _i4_alwayson = True

    def ui(self, is_img2img):
      import gradio as gr
      try:
        from pi_ideogram_lib.quant_registry import all_quant_ids
        _quants = ["auto"] + all_quant_ids()
      except Exception:
        _quants = ["auto"]
      # Visible when the model is loaded AND when the dropdown already points
      # at an Ideogram 4 file (persisted selection at boot, or a change event
      # that has not fired yet).
      _visible = is_ideogram4_selected() or _selected_ci_is_i4()
      # UI defaults come from pi_ideogram_lib.settings (single owner of the
      # UI<->settings mapping); config.json feeds settings, engine never reads
      # config values itself for UI defaults.
      _def = pi_settings.ui_defaults()
      _preset_choices = list(pi_settings.PRESET_CHOICES)
      _preset_val = _def["preset"]
      if _preset_val not in _preset_choices:
        _preset_val = "V4_DEFAULT_20"
        _def["preset"] = _preset_val
      # ---------------------------------------------------------------- #
      # LAYOUT. Five controls sat in one flat row, so "which preset" (used
      # every generation) had the same visual weight as "quant override"
      # (used approximately never), and the panel read as five equal
      # decisions rather than one decision plus four escape hatches.
      #
      # Now: the preset and what it actually does are the whole top level;
      # everything else is behind "Advanced". The RETURNED tuple is
      # unchanged - it is the script-args contract (UI_KEYS parity), so the
      # order of the 10 components below must not move even though their
      # on-screen position has.
      # ---------------------------------------------------------------- #
      _labels = {'V4_DEFAULT_20': 'Balanced · recommended',
                 'V4_QUALITY_48': 'Detail · quality recipe',
                 'V4_TURBO_12': 'Quick · full CFG',
                 'TurboTime-4': 'Turbo · requires TurboTime adapter',
                 'Instant-8': 'Instant · distilled recipe'}
      with gr.Accordion("Ideogram 4", open=False, visible=_visible,
                        elem_classes=['pi-i4-panel']) as i4_accordion:
        gr.Markdown(value=_quickstart())
        if is_img2img:
          gr.Markdown(
            "**Experimental image-to-image:** use Forge's standard image upload "
            "and **Denoising strength** controls above, then choose a recipe here. "
            "Lower denoising keeps more of the source image; higher denoising changes more. "
            "Instruction editing and inpainting are not supported by this panel.")
        with gr.Row(elem_classes=['pi-i4-recipe']):
          preset_dd = gr.Dropdown(
            choices=[(_labels.get(p, p), p) for p in _preset_choices],
            value=_preset_val,
            label="Generation recipe",
            scale=2,
            info="Balanced for everyday use. Detail for longer runs. Turbo for fast previews.")
        with gr.Accordion("Technical details", open=False):
          recipe_md = gr.Markdown(value=_recipe_line(_preset_val))
          gr.Markdown("Use Forge's size, seed, batch and Steps controls. Named recipes own their CFG schedule; "
                      "distilled recipes also own their step count. Forge's sampler and scheduler selectors do not apply here.")

        with gr.Tabs(elem_classes=["pi-i4-tools"]):
          with gr.Tab("Style"):
            # ---- adapter picker ------------------------------------------
            # The bypass dropdown above deliberately lists ONLY filter-bypass
            # adapters. The TurboTime and uncond-replacement LoRAs are excluded
            # from it because they are not bypasses - which left no way to use
            # them except typing a <lora:...> tag. This lists every Ideogram 4
            # adapter found (subfolders included) with its ROLE, so picking one
            # is the same as typing the tag, and the role makes clear what it
            # will actually do.
            from pi_ideogram_lib.detect import (ADAPTER_NONE_LABEL, adapter_choices,
                                                adapter_name_of)
            try:
              _adapters = adapter_choices(_models_dirs())
            except Exception:
              _adapters = []
            _adapter_choices = [ADAPTER_NONE_LABEL] + _adapters
            _adapter_val = _def.get("adapter_choice") or ADAPTER_NONE_LABEL
            if _adapter_val not in _adapter_choices:
              _adapter_val = ADAPTER_NONE_LABEL

            gr.Markdown(value=(
              "Choose a style or character adapter here. You can combine more using `<lora:name:strength>` in your prompt."
              if _adapters else
              "**Adapter** - no Ideogram 4 adapters found. Put them in "
              "`models/Lora/IDEOGRAM LORA` (subfolders are searched too)."))

            adapter_dd = gr.Dropdown(
              choices=_adapter_choices, value=_adapter_val,
              label="Adapter",
              interactive=bool(_adapters),
              info="Applied on top of any <lora:...> tags in your prompt.")
            adapter_str = gr.Slider(
              minimum=0.0, maximum=1.5, step=0.05,
              value=float(_def.get("adapter_strength", 1.0)),
              label="Adapter strength",
              visible=(_adapter_val != ADAPTER_NONE_LABEL),
              info="1.0 is the trained strength for character and turbo adapters.")

            def _toggle_adapter_strength(name):
              return gr.update(visible=str(name) != ADAPTER_NONE_LABEL)

            adapter_dd.change(fn=_toggle_adapter_strength, inputs=[adapter_dd],
                              outputs=[adapter_str], show_progress=False)

          with gr.Tab("Filter bypass"):
            # TWO controls, on purpose. This was eight (on/off, a master mix and
            # six "advanced" levers) for something most people set once.
            #
            # It is also the more correct shape: a bypass LoRA is a weight delta
            # applied for a single step, so it does NOT touch the sampler, the
            # scheduler or the step count - the official recipe stays official
            # while the bypass is in use. The old master slider drove sigma
            # smoothing, which inserted steps ("Default 20" really sampled 22).
            # The remaining levers still live in bypass.py for config.json users.
            _bypass_choices = [pi_settings.BYPASS_NONE_LABEL]
            try:
              from pi_ideogram_lib.detect import bypass_lora_choices
              _bypass_choices += bypass_lora_choices(_models_dirs())
            except Exception:
              pass
            # A persisted name that is no longer a valid choice falls back to
            # "None" - it must NOT be appended back into the list. Doing that
            # re-offered exactly the entries the choice list had just excluded
            # (the uncond-replacement and TurboTime adapters, which are not
            # bypasses), so a stale config.json could keep an invalid selection
            # alive across every restart.
            _bypass_val = _def.get("bypass_lora_name") or pi_settings.BYPASS_NONE_LABEL
            if _bypass_val not in _bypass_choices:
              _bypass_val = pi_settings.BYPASS_NONE_LABEL
            _has_bypass = len(_bypass_choices) > 1

            # An empty dropdown reads as a broken feature. Say plainly that none
            # is installed, and where to put one - the answer to "why is this
            # list empty" should be in the panel, not in a console log.
            bypass_help_md = gr.Markdown(value=_bypass_help(_has_bypass))

            bypass_lora_dd = gr.Dropdown(
              choices=_bypass_choices, value=_bypass_val,
              label="Bypass adapter",
              interactive=_has_bypass,
              info="None turns the adapter off.")
            bypass_lora_str = gr.Slider(
              minimum=-1.0, maximum=0.25, step=0.05,
              value=_def.get("bypass_lora_strength", -0.25),
              label="Strength",
              visible=(_bypass_val != pi_settings.BYPASS_NONE_LABEL),
              info="Gray starting point: −0.25. Zero disables it. More negative is not always better; image details can change.")
            bypass_status_md = gr.Markdown(value=_bypass_status(
                _bypass_val, _def.get("bypass_lora_strength", -0.25), _def["json_pass_through"]),
                elem_classes=['pi-i4-status'])
            with gr.Row():
              bypass_recommended = gr.Button("Use Gray starting point", size="sm")
              bypass_combo = gr.Button("Gray + guidance LoRA (1.0)", size="sm")
              bypass_off = gr.Button("Turn off", size="sm")
              bypass_refresh = gr.Button("Refresh adapters", size="sm")

            def _refresh_bypasses(current):
              from pi_ideogram_lib.detect import bypass_lora_choices
              choices = [pi_settings.BYPASS_NONE_LABEL] + bypass_lora_choices(_models_dirs())
              value = current if current in choices else pi_settings.BYPASS_NONE_LABEL
              return gr.update(choices=choices, value=value, interactive=len(choices) > 1), _bypass_help(len(choices) > 1)

            def _recommended_bypass():
              from pi_ideogram_lib.detect import bypass_lora_choices
              choices = [pi_settings.BYPASS_NONE_LABEL] + bypass_lora_choices(_models_dirs())
              gray = next((n for n in choices if Path(n).stem.lower() == "gray_000002000"), None)
              if gray is None:
                gr.Warning("Install Gray_000002000.safetensors from Model Setup first. Your current settings were kept.")
                return gr.update(), gr.update()
              return gr.update(choices=choices, value=gray, interactive=True), gr.update(value=-0.25, visible=True)

            bypass_recommended.click(fn=_recommended_bypass, inputs=[],
                                     outputs=[bypass_lora_dd, bypass_lora_str], show_progress=False)
            def _combined_bypass():
              from pi_ideogram_forge.lora import find_uncond_replacement_lora
              if find_uncond_replacement_lora(_models_dirs()) is None:
                gr.Warning("Install ideogram_4_unconditional_lora_r16.safetensors in Model Setup first.")
                return gr.update(), gr.update(), gr.update()
              adapter, strength = _recommended_bypass()
              if 'value' not in adapter:
                return adapter, strength, gr.update()
              return adapter, strength, gr.update(value="Ostris uncond LoRA")
            bypass_off.click(fn=lambda: pi_settings.BYPASS_NONE_LABEL, inputs=[],
                             outputs=[bypass_lora_dd], show_progress=False)
            bypass_refresh.click(fn=_refresh_bypasses, inputs=[bypass_lora_dd],
                                 outputs=[bypass_lora_dd, bypass_help_md], show_progress=False)

            # The strength slider is meaningless with no LoRA selected, so it is
            # only shown once one is.
            def _toggle_strength(name):
              return gr.update(visible=str(name) != pi_settings.BYPASS_NONE_LABEL)

            bypass_lora_dd.change(fn=_toggle_strength, inputs=[bypass_lora_dd],
                                  outputs=[bypass_lora_str], show_progress=False)

          with gr.Tab("Performance"):
            from pi_ideogram_lib.attention import attention_status
            gr.Markdown(value="Memory is managed automatically. Acceleration trades image fidelity for speed.")
            with gr.Row():
              quant_dd = gr.Dropdown(choices=_quants, value=_def["quant_override"],
                                     label="Checkpoint format",
                                     info="Auto reads the file. This does not convert or re-quantize your weights.")
              uncond_dd = gr.Radio(choices=["Auto", "Full transformer", "Ostris uncond LoRA"],
                                   value=_def["uncond_mode"],
                                   label="Guidance model",
                                   info="Full transformer preserves the original pair. The replacement adapter uses less memory.")
            with gr.Row():
              json_pt = gr.Checkbox(value=_def["json_pass_through"],
                                    label="Pass-through raw prompt (no JSON wrap)",
                                    info="Send the prompt box verbatim. Turns OFF the density guard.")
              upsampling_cb = gr.Checkbox(value=_def["prompt_upsampling"],
                                          label="Local prompt upsampling",
                                          info="Expands sparse prompts locally (no LLM, no cloud)")
            with gr.Row():
              _acc = _def.get('spectrum_enabled', False)
              spectrum_cb = gr.Dropdown(
                  choices=[('Off · full computation', 'off'), ('Spectrum · faster, approximate', 'spectrum'),
                           ('First Block Cache · experimental', 'first_block')],
                  value='first_block' if _acc == 'first_block' else ('spectrum' if _acc else 'off'),
                  label="Extra acceleration",
                  info="Choose one. Both change image details. First Block Cache reuses similar block results; speed depends on the prompt. Off preserves full computation.")

          bypass_combo.click(fn=_combined_bypass, inputs=[],
                             outputs=[bypass_lora_dd, bypass_lora_str, uncond_dd], show_progress=False)
          for control in (bypass_lora_dd, bypass_lora_str, json_pt):
            control.change(fn=_bypass_status, inputs=[bypass_lora_dd, bypass_lora_str, json_pt],
                           outputs=[bypass_status_md], show_progress=False)
          preset_dd.change(fn=_recipe_line, inputs=[preset_dd], outputs=[recipe_md],
                           show_progress=False)

          # ---- spec F: model setup / variant picker --------------------------
          # Every component here is intentionally NOT in the returned list: the
          # 10 returned controls stay the script-args contract (UI_KEYS parity,
          # see ARCHITECTURE.md). Buttons fire downloads on EXPLICIT clicks only.
          try:
            from pi_ideogram_assets.downloader import SOURCED_QUANTS
          except Exception:
            SOURCED_QUANTS = ()
          _picker_cfg_data = _picker_cfg()
          _quant = _picker_quant(_picker_cfg_data)
          _v_choices = [q for q in SOURCED_QUANTS]
          if _quant not in _v_choices:
            _v_choices.insert(0, _quant)
          with gr.Tab("Model Setup"):
            _sel_ci = _resolve_ci_for_name(_selected_checkpoint_name())
            _detect_value = ("No Ideogram 4 checkpoint selected."
                             if not (_sel_ci and _ci_is_ideogram4(_sel_ci))
                             else _routing_markdown(path=getattr(_sel_ci, "filename", None),
                                                    cfg=_picker_cfg_data))
            detect_md = gr.Markdown(value=_detect_value)
            # (the static "recommended variant" card used to sit here: it had no
            # event wiring and repeated what the picker listing below already
            # says, so it was three status blocks describing one thing)
            picker_md = gr.Markdown(value=_picker_markdown(_models_dirs(), _quant, _picker_cfg_data))
            variant_dd = gr.Dropdown(
              choices=_v_choices, value=_quant, label="Checkpoint variant",
              info="Downloaded files appear in the checkpoint dropdown as 'Ideogram4 - <name>'. "
                   "Recommended for your GPU by default; your choice is remembered.")
            # NO DOWNLOAD BUTTONS, NO API KEY.
            #
            # The buttons that used to sit here fetched assets over the network,
            # and the Civitai one needed an API token pasted into a password box
            # and stored in config.json - the file most likely to end up in a
            # screenshot or a bug report. The gray-bypass LoRA is creator-gated,
            # so without a key the button could not work at all; with one, this
            # extension was asking for a credential to do a job the browser
            # already does better (signed in, resumable, shows the licence gate).
            #
            # Links and destinations instead. Nothing here touches the network.
            from pi_ideogram_assets.links import markdown as _links_markdown
            from pi_ideogram_assets.setup_ui import build as _download_setup
            _download_setup(variant_dd, _models_dirs)
            gr.Markdown(value=_links_markdown())

            def _variant_changed(q):
              return _on_variant_changed(str(q or _quant), _models_dirs())

            variant_dd.change(fn=_variant_changed, inputs=[variant_dd], outputs=[picker_md])
      # Visibility follows the checkpoint dropdown - but that dropdown does not
      # exist yet. Forge builds script UI (modules/ui.py -> scripts_txt2img
      # .setup_ui) long before it builds the quicksettings row that creates
      # ui_checkpoint (ui_settings.add_quicksettings -> main_entry
      # .make_checkpoint_manager_ui), and main_entry only ANNOTATES the name at
      # module level. So importing it here always raised, the bare except
      # swallowed it, and the panel was frozen at the state it was built with:
      # selecting an Ideogram 4 checkpoint did nothing until a page reload.
      # Register the pair instead; install_visibility_hook() below wires it the
      # moment the real dropdown is created.
      _I4_PANELS.append((i4_accordion, detect_md))
      # MUST match pi_ideogram_lib.settings.UI_KEYS, order included.
      return [preset_dd, quant_dd, uncond_dd, json_pt, upsampling_cb,
              bypass_lora_dd, bypass_lora_str, adapter_dd, adapter_str,
              spectrum_cb]
    # NOTE: no process() override. Always-visible process() only runs inside
    # processing.process_images, which the takeover skips for Ideogram 4;
    # accordion values are captured in run._wrapped_run via _resolve_run_opts.

  install_wrapper()
  install_dropdown_hook()
  install_visibility_hook()
