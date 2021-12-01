_LIGHTNING_MODULES = {}
def register_lightning_module(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _LIGHTNING_MODULES:
      raise ValueError(f'Already registered model with name: {local_name}')
    _LIGHTNING_MODULES[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_lightning_module_by_name(name):
  return _LIGHTNING_MODULES[name]

def create_lightning_module(config, checkpoint_path=None):
  lightning_module = get_lightning_module_by_name(config.training.lightning_module)(config)
  if checkpoint_path:
    lightning_module = lightning_module.load_from_checkpoint(checkpoint_path)
  return lightning_module