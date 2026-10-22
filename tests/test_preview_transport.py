"""Exercise the installed Forge progress endpoint's actual image gate, offline."""
import ast
import base64
import io
from pathlib import Path
from types import SimpleNamespace
import time

from PIL import Image
from pydantic import BaseModel, Field

from pi_ideogram_forge.preview import native_preview_session, publish_start, publish_final


class Options:
    def __init__(self):
        self.data = dict(live_previews_enable=False, live_previews_image_format='png',
                         show_progress_every_n_steps=-1)

    def __getattr__(self, key):
        return self.data[key]


def test_installed_forge_endpoint_sends_developing_and_exact_final_pixels():
    host = Path(__file__).resolve().parents[3]
    parsed = ast.parse((host / 'modules/progress.py').read_text(encoding='utf-8'))
    names = {'ProgressRequest', 'ProgressResponse', 'progressapi'}
    nodes = [node for node in parsed.body if getattr(node, 'name', None) in names]
    assert len(nodes) == 3, 'Installed Forge progress API changed'
    state = SimpleNamespace(job_count=1, job_no=0, sampling_steps=4,
                            sampling_step=0, time_start=time.time(), textinfo='Ideogram 4',
                            id_live_preview=0, current_image=None)
    options = Options()
    ns = dict(BaseModel=BaseModel, Field=Field, io=io, base64=base64, time=time,
              shared=SimpleNamespace(state=state), opts=options,
              current_task='task(test)', pending_tasks={}, finished_tasks=[])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'installed_progress', 'exec'), ns)
    request = ns['ProgressRequest'](id_task='task(test)', id_live_preview=-1)
    endpoint = ns['progressapi']
    publish_start(state, 512, 512)
    assert endpoint(request).live_preview is None
    with native_preview_session(options):
        first = endpoint(request)
        assert first.live_preview.startswith('data:image/png;base64,')
        request.id_live_preview = first.id_live_preview
        assert endpoint(request).live_preview is None  # no duplicate frame
        final = Image.new('RGB', (512, 512), '#238647')
        publish_final(state, final, 4)
        response = endpoint(request)
        decoded = Image.open(io.BytesIO(base64.b64decode(response.live_preview.split(',')[1])))
        assert decoded.size == final.size
        assert decoded.tobytes() == final.tobytes()
        assert response.progress == 1.0
    assert not options.live_previews_enable
    assert options.show_progress_every_n_steps == -1
    assert endpoint(request).live_preview is None
