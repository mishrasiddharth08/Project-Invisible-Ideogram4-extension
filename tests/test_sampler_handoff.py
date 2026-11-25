from types import SimpleNamespace

import pytest

from pi_ideogram_forge.contract import normalize_ui_sampler, validate_request
from pi_ideogram_forge.ui_defaults import _sampler_value, _selection_values


def request(**updates):
    values = dict(sampler_name='iPNDM', scheduler='Karras', prompt='scene',
                  negative_prompt='', width=1024, height=1024,
                  extra_generation_params={})
    values.update(updates)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('sampler,scheduler', [
    ('Euler', 'Simple'), ('iPNDM', 'Karras'), ('DPM++ 2M', 'SGM Uniform')])
def test_ui_and_api_sampler_scheduler_intent_is_not_rewritten(sampler, scheduler):
    p = request(sampler_name=sampler, scheduler=scheduler)
    assert normalize_ui_sampler(p) is None
    validate_request(p)
    assert (p.sampler_name, p.scheduler) == (sampler, scheduler)


def test_legacy_ui_seams_do_not_force_sampler_values():
    assert _selection_values('Ideogram4 - model') == (None, None)
    assert _sampler_value('Ideogram4 - model', 'iPNDM') is None
