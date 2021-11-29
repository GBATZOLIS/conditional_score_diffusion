_CALLBACKS = {}
def register_callback(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CALLBACKS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CALLBACKS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_callback_by_name(name):
    return _CALLBACKS[name]

def get_callbacks(config):
    callbacks=[get_callback_by_name('ema')()] #check if this works for testing as well.

    if config.eval.callback is not None:
      callbacks.append(get_callback_by_name(config.eval.callback)(show_evolution=False, 
                                                                  eval_config=config.eval, 
                                                                  data_config=config.data,
                                                                  approach = config.training.conditioning_approach))
    else:
      callbacks.append(get_callback_by_name(config.training.visualization_callback)(show_evolution=config.training.show_evolution))

    if config.training.lightning_module in ['conditional_decreasing_variance','haar_conditional_decreasing_variance'] :
      callbacks.append(get_callback_by_name('decreasing_variance_configuration')(config))
    else:
      callbacks.append(get_callback_by_name('configuration')())

    return callbacks

  