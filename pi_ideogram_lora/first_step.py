"""Temporary adapters are scoped to each image, never to an entire batch."""
from .runtime_lora import RuntimeLoraAdapter


class FirstStepAdapters:
    def __init__(self, models, state_dict, strength):
        self.models = [model for model in models if model is not None]
        self.state_dict = state_dict
        self.strength = float(strength)
        self.adapters = []

    def begin(self):
        self.finish()
        if not self.strength:
            return
        try:
            for model in self.models:
                adapter = RuntimeLoraAdapter(model, self.state_dict, self.strength, 'gray-bypass')
                # Own it before install, so partial installs also get rolled back.
                self.adapters.append(adapter)
                if adapter.install() == 0:
                    raise ValueError('Gray adapter matched no modules in the selected transformer')
                adapter.enabled = True
        except Exception as exc:
            self.finish()
            raise ValueError(f'Gray bypass could not be applied: {exc}') from exc

    def finish(self):
        held, self.adapters = self.adapters, []
        errors = []
        for adapter in held:
            adapter.enabled = False
            try:
                adapter.remove()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('Could not remove Gray adapter hooks: ' + '; '.join(errors))
