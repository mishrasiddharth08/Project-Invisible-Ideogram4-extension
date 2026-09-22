"""Manual-first setup; only selected files download on an explicit click."""
from pathlib import Path
from . import downloader as dl
MANUAL = 'Manual download (recommended)'
AUTOMATIC = 'Download selected files for me'
ASSETS = ('checkpoint:cond', 'checkpoint:uncond', 'te', 'vae', 'adapter:turbotime', 'adapter:uncond')

def choices(quant):
    rows = []
    for asset in ASSETS:
        if asset.startswith('adapter:'):
            filename = (dl.HF_TURBOTIME if asset.endswith('turbotime') else dl.HF_UNCOND)[1]
        else:
            source = dl._asset_source(asset, quant)
            if source is None:
                continue
            filename = Path(source[1]).name
        rows.append((f'{dl.ASSET_LABELS[asset]} — {filename}', asset))
    return rows

def download_selected(mode, selected, quant, dirs):
    selected = list(dict.fromkeys(selected or []))
    if mode != AUTOMATIC or not selected or not set(selected).issubset({v for _, v in choices(quant)}):
        yield 'No downloads started. Choose automatic mode and select files, or use the manual links below.'
        return
    messages = []
    for asset in selected:
        label = dl.ASSET_LABELS[asset]
        yield '\n\n'.join(messages + [f'Downloading/checking {label}…'])
        try:
            path = dl.fetch_asset(asset, quant, dirs)
            messages.append(f'{label}: {path if path else "already installed"}')
        except Exception as exc:
            messages.append(f'{label}: failed — {exc}. Use the manual link below.')
        yield '\n\n'.join(messages)
    yield '\n\n'.join(messages + ['Finished. Refresh Forge’s model list to see installed files.'])

def build(variant, models_dirs):
    import gradio as gr
    mode = gr.Radio([MANUAL, AUTOMATIC], value=MANUAL, label='How would you like to get model files?')
    gr.Markdown('Required: **one checkpoint + text encoder + VAE**. Adapters are optional. '
                'Gray requires a manual creator download. Nothing downloads on startup or Generate.')
    with gr.Column(visible=False) as automatic:
        selected = gr.CheckboxGroup(choices(variant.value), value=[], label='Choose files to download')
        gr.Markdown('These can be large downloads. Only selected files are fetched; installed files are reused.')
        button = gr.Button('Download selected files', variant='primary')
        status = gr.Markdown()
    mode.change(lambda value: gr.update(visible=value == AUTOMATIC), inputs=[mode], outputs=[automatic], show_progress=False)
    variant.change(lambda value: gr.update(choices=choices(value), value=[]), inputs=[variant], outputs=[selected], show_progress=False)
    def run(value, selection, quant):
        yield from download_selected(value, selection, quant, models_dirs())
    button.click(run, inputs=[mode, selected, variant], outputs=[status])
