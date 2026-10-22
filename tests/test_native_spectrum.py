# requires: cpu
import sys
from pathlib import Path

import pytest
import torch


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
FORGE_ROOT = EXTENSION_ROOT.parents[1]
for path in (EXTENSION_ROOT, FORGE_ROOT, FORGE_ROOT / "modules_forge" / "packages"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_ideogram_lib.native_spectrum import ForgeSpectrum


FAST = dict(m=1, w=0.0, warmup_steps=0, window_size=2,
            flex_window=0.0, stop_caching_step=1.0)


@pytest.fixture
def spectrum_factory():
    sessions = []

    def create(**options):
        try:
            session = ForgeSpectrum(8, torch.device("cpu"), options)
        except RuntimeError as error:
            pytest.skip(f"installed Forge Spectrum backend unavailable: {error}")
        expected = (FORGE_ROOT / "extensions-builtin" / "sd_forge_spectrum" /
                    "lib_spectrum" / "__init__.py").resolve()
        assert Path(session.source_path).resolve() == expected
        assert type(session._anchor).__module__ == "backend.patcher.unet"
        assert session._node.__module__ == "lib_spectrum"
        sessions.append(session)
        return session

    yield create
    for session in sessions:
        session.close()


def _x(batch=2):
    return torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 1, 2, 2)


def _run(session, branch, x, values, forward):
    return [session.evaluate(branch, x, torch.full((x.shape[0],), value), forward)
            for value in values]


def test_window_one_is_exact_and_never_skips(spectrum_factory):
    session = spectrum_factory(**(FAST | {"window_size": 1}))
    x = _x()
    seen = []

    def forward(x_real, t_real, indices):
        seen.append((t_real.clone(), indices.clone()))
        return x_real + t_real[:, None, None, None]

    values = (0.1, 0.2, 0.3, 0.4, 0.5)
    outputs = _run(session, 0, x, values, forward)
    for value, output in zip(values, outputs):
        torch.testing.assert_close(output, x + value, rtol=0, atol=0)
    assert len(seen) == len(values)
    assert session.real_calls == len(values)
    assert session.forecast_calls == session.forecast_items == 0


def test_enabled_native_spectrum_skips_actual_forward(spectrum_factory):
    session = spectrum_factory(**FAST)
    calls = []

    def forward(x_real, t_real, indices):
        calls.append(indices.clone())
        return torch.full_like(x_real, 3.0)

    outputs = _run(session, 0, _x(), (0.1, 0.2, 0.3, 0.4), forward)
    assert len(calls) == 3
    assert session.real_calls == 3
    assert session.forecast_calls == 1
    assert session.real_items == 6
    assert session.forecast_items == 2
    assert torch.isfinite(outputs[-1]).all()


def test_branches_keep_independent_native_histories(spectrum_factory):
    session = spectrum_factory(**FAST)
    x = torch.zeros_like(_x())

    def low(x_real, _t, _indices):
        return torch.full_like(x_real, 1.0)

    def high(x_real, _t, _indices):
        return torch.full_like(x_real, 10.0)

    _run(session, 0, x, (0.1, 0.2, 0.3), low)
    _run(session, 1, x, (0.1, 0.2, 0.3), high)
    low_forecast = session.evaluate(0, x, torch.full((2,), 0.4), low)
    high_forecast = session.evaluate(1, x, torch.full((2,), 0.4), high)
    assert low_forecast.mean() < 3
    assert high_forecast.mean() > 7
    assert session.forecast_calls == 2


def test_increasing_original_time_is_passed_unchanged_and_does_not_reset(spectrum_factory):
    session = spectrum_factory(**FAST)
    seen = []

    def forward(x_real, t_real, indices):
        seen.append((t_real.clone(), indices.clone()))
        return x_real + 1

    _run(session, 0, _x(), (0.125, 0.25, 0.375, 0.5), forward)
    assert len(seen) == 3  # fourth call is forecast; wrong timestep direction makes all real
    for expected, (actual_t, indices) in zip((0.125, 0.25, 0.375), seen):
        torch.testing.assert_close(actual_t, torch.full((2,), expected), rtol=0, atol=0)
        assert indices.tolist() == [0, 1]


def test_reset_discards_history_after_weight_change(spectrum_factory):
    session = spectrum_factory(**FAST)
    x = torch.zeros_like(_x())
    weight = [2.0]

    def forward(x_real, _t, _indices):
        return torch.full_like(x_real, weight[0])

    old = _run(session, 0, x, (0.1, 0.2, 0.3, 0.4), forward)[-1]
    weight[0] = 9.0
    session.reset()
    new = _run(session, 0, x, (0.5, 0.6, 0.7, 0.8), forward)[-1]
    assert old.mean() < 4
    assert new.mean() > 7


def test_partial_native_batch_mask_preserves_original_indices(spectrum_factory):
    session = spectrum_factory(**FAST)
    x = torch.zeros_like(_x())

    def warm(x_real, _t, indices):
        return indices[:, None, None, None].expand_as(x_real).to(x_real)

    _run(session, 0, x, (0.1, 0.2, 0.3), warm)

    wrapper = session._patchers[0].model_options["model_function_wrapper"]
    state = next(cell.cell_contents for cell in wrapper.__closure__
                 if hasattr(cell.cell_contents, "num_cached") and
                 hasattr(cell.cell_contents, "forecasters"))
    state.num_cached[1] = 1  # item 0 forecasts; item 1 performs the real call
    seen = []

    def changed(x_real, t_real, indices):
        seen.append((x_real.clone(), t_real.clone(), indices.clone()))
        return torch.full_like(x_real, 100.0) + indices[:, None, None, None]

    output = session.evaluate(0, x, torch.full((2,), 0.4), changed)
    assert len(seen) == 1
    real_x, real_t, indices = seen[0]
    assert real_x.shape[0] == 1
    assert indices.tolist() == [1]
    torch.testing.assert_close(real_t, torch.tensor([0.4]), rtol=0, atol=0)
    torch.testing.assert_close(output[1], torch.full_like(output[1], 101.0))
    assert session.forecast_items == 1


def test_cfg_branch_gap_invalidates_old_native_history(spectrum_factory):
    session = spectrum_factory(**FAST)
    x = _x(batch=1)
    for step in range(3):
        session.evaluate(1, x, torch.tensor([step / 10]),
                         lambda x, t, indices: torch.ones_like(x), step_index=step)
    before = session.real_calls
    changed = session.evaluate(1, x, torch.tensor([.6]),
                               lambda x, t, indices: torch.full_like(x, 99.), step_index=6)
    assert session.real_calls == before + 1
    torch.testing.assert_close(changed, torch.full_like(x, 99.), rtol=0, atol=0)
