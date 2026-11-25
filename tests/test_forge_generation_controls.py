"""Drive the real Forge batch loop with a CPU-only callback pipeline."""
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from pi_ideogram_forge.generate import _run_sampling
from pi_ideogram_lib.bypass import recipe


def run_job(*, action=None, enabled=True, cadence=1, batches=2, batch_size=2,
            before=None, notice=None, gray=False):
    state = SimpleNamespace(interrupted=False, skipped=False, sampling_step=0,
                            id_live_preview=0, current_image=None)
    options = SimpleNamespace(live_previews_enable=enabled,
                              show_progress_every_n_steps=cadence)
    shared = SimpleNamespace(state=state, opts=options, total_tqdm=Mock(),
                             cmd_opts=SimpleNamespace(disable_console_progressbars=True))
    p = SimpleNamespace(prompt='test', seed=7, n_iter=batches, batch_size=batch_size,
                        extra_generation_params={}, save_samples=lambda: False)
    calls, decoded, saved, frames = [], [], [], []
    state.assign_current_image = lambda image: frames.append(image)
    final = Image.new('RGB', (8, 8), 'red')

    class Pipe:
        conditional_transformer = None
        unconditional_transformer = None
        uncond_adapter = None
        _native_spectrum_stats = dict(real_calls=3, forecast_calls=2, notice=notice)

        def decode_preview(self, latent, **kwargs):
            decoded.append((len(calls), state.sampling_step))
            return final

        def __call__(self, caption, **kwargs):
            calls.append(kwargs['seed'])
            if isinstance(notice, list):
                self._native_spectrum_stats = dict(notice=notice[len(calls) - 1])
            assert (kwargs['preview_callback'] is None) == (
                not options.live_previews_enable or options.show_progress_every_n_steps < 0)
            kwargs['on_actual_steps'](6)
            for step in range(6):
                if action:
                    action(state, options, len(calls), step)
                kwargs['step_callback'](step, 6)
                if kwargs['preview_callback'] is not None:
                    kwargs['preview_callback'](None)
            return [final]

    if before:
        before(state)
    result = _run_sampling(
        p, Pipe(), 'test', 8, 8, 6, (1.,) * 6, 0., 1.75,
        'custom', 'fp8', [], recipe(0.), Path('gray.safetensors') if gray else None,
        .25 if gray else 0., False, False, False,
        .6, 'raw', Path('test.safetensors'),
        native_spectrum={} if notice else None,
        _FORGE_OK=True, shared=shared,
        Processed=lambda p, images, **kw: SimpleNamespace(images=images, **kw),
        forge_images=SimpleNamespace(save_image=lambda *a, **kw: saved.append(a)),
        lora_logs=[], active_loras=None)
    return result, state, calls, decoded, frames, p


@pytest.mark.parametrize('batch_size', [1, 2])
def test_skip_discards_current_image_and_continues_across_iterations(batch_size):
    def skip(state, options, call, step):
        if call == 1 and step == 1:
            state.skipped = True
    result, state, calls, *_ = run_job(action=skip, batch_size=batch_size)
    assert calls == list(range(7, 7 + 2 * batch_size))
    assert result.all_seeds == calls[1:]
    assert result.seed == calls[1]
    assert len(result.images) == len(calls) - 1
    assert not state.skipped and not state.interrupted


def test_stop_wins_over_skip_and_preserves_completed_images():
    def stop(state, options, call, step):
        if call == 2 and step == 1:
            state.interrupted = state.skipped = True
    result, state, calls, *_ = run_job(action=stop)
    assert calls == [7, 8]
    assert result.all_seeds == [7]
    assert len(result.images) == 1
    assert state.interrupted


@pytest.mark.parametrize('flag', ['interrupted', 'stopping_generation'])
def test_stop_before_sampling_does_not_start_pipeline(flag):
    result, _, calls, *_ = run_job(before=lambda state: setattr(state, flag, True))
    assert calls == [] and result.images == []


@pytest.mark.parametrize('batch_size', [1, 2])
def test_soft_stop_finishes_current_image_and_starts_no_more(batch_size):
    def soft_stop(state, options, call, step):
        if call == 1 and step == 1:
            state.stopping_generation = True
    result, state, calls, decoded, frames, _ = run_job(
        action=soft_stop, batch_size=batch_size)
    assert calls == [7]
    assert result.all_seeds == [7] and len(result.images) == 1
    assert state.sampling_step == 6
    assert frames[-1] is result.images[0]
    assert not state.interrupted


@pytest.mark.parametrize('stop', [False, True])
def test_cancel_restores_first_step_adapter(monkeypatch, stop):
    adapter = Mock()
    monkeypatch.setattr('pi_ideogram_lora.first_step.FirstStepAdapters', lambda *a: adapter)
    monkeypatch.setattr('pi_ideogram_lora.ideogram4_lora.load_lora_state_dict', lambda *a: {})
    def cancel(state, options, call, step):
        if call == 1 and step == 0:
            if stop:
                state.interrupted = True
            else:
                state.skipped = True
    run_job(action=cancel, gray=True, batches=1, batch_size=2)
    assert adapter.begin.call_count == (1 if stop else 2)
    assert adapter.finish.call_count == adapter.begin.call_count


def test_pending_skip_consumes_one_image_only():
    result, state, calls, *_ = run_job(before=lambda state: setattr(state, 'skipped', True))
    assert calls == [8, 9, 10]
    assert result.all_seeds == calls and not state.skipped


@pytest.mark.parametrize('enabled,cadence,steps,frame_count', [
    (False, 1, [], 0), (True, -1, [], 1),
    (True, 0, [1, 2, 3, 4, 5], 7),
    (True, 2, [2, 4], 4), (True, 3, [3], 3),
])
def test_preview_preferences_and_exact_step_cadence(enabled, cadence, steps, frame_count):
    result, _, _, decoded, frames, _ = run_job(
        enabled=enabled, cadence=cadence, batches=1, batch_size=1)
    assert [step for _, step in decoded] == steps
    assert len(frames) == frame_count
    if enabled:
        assert frames[-1] is result.images[-1]


def test_preview_changes_are_observed_during_sampling():
    def disable(state, options, call, step):
        if step == 2:
            options.live_previews_enable = False
    _, _, _, decoded, frames, _ = run_job(action=disable, batches=1, batch_size=1)
    assert [step for _, step in decoded] == [1, 2]
    assert len(frames) == 3  # start canvas and two intermediate frames


def test_spectrum_notice_is_in_infotext_and_extra_parameters():
    notice = 'Forecast unavailable; using real forwards'
    result, *_, p = run_job(notice=notice, batches=1, batch_size=1)
    assert 'Spectrum notice: ' + notice in result.infotexts[0]
    assert p.extra_generation_params['Spectrum notice'] == notice


@pytest.mark.parametrize('next_notice', [None, ''])
def test_spectrum_notice_is_cleared_for_following_image(next_notice):
    notice = 'Forecast unavailable; using real forwards'
    result, *_, p = run_job(notice=[notice, next_notice], batches=1, batch_size=2)
    assert 'Spectrum notice: ' + notice in result.infotexts[0]
    assert 'Spectrum notice:' not in result.infotexts[1]
    assert 'Spectrum notice' not in p.extra_generation_params
