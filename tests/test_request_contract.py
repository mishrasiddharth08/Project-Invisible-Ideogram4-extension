# requires: cpu
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi_ideogram_forge.contract import validate_request, validate_native_spectrum


def request(**values):
    defaults = dict(prompt='scene', negative_prompt='', sampler_name='Euler',
                    width=512, height=512, subseed_strength=0, enable_hr=False)
    defaults.update(values)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize('changes', [dict(prompt=['one', 'two']), dict(seed=[1, 2]),
    dict(negative_prompt='blur'), dict(enable_hr=True), dict(subseed_strength=.2),
    dict(sampler_name='DPM++ 2M'), dict(width=513), dict(height=4096)])
def test_unsupported_request_fails_before_loading(changes):
    with pytest.raises(ValueError):
        validate_request(request(**changes))


def test_native_txt2img_and_img2img_inputs_are_accepted():
    validate_request(request())
    validate_request(request(init_images=[object()], denoising_strength=.5))


def test_builtin_spectrum_settings_reach_the_request():
    script = type('SpectrumForForge', (), {'args_from': 1})()
    runner = SimpleNamespace(alwayson_scripts=[script])
    enabled = validate_native_spectrum(runner, request(
        script_args=(False, True, .25, 6, .5, 2, 0., 6, .9)))
    assert enabled['w'] == .25 and enabled['m'] == 6
    assert validate_native_spectrum(runner, request(script_args=(False, False))) is None
