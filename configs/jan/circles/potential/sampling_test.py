from configs.jan.circles.potential.circles_potential import get_config as get_config_potential

def get_config():
    config = get_config_potential()
    
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor =  'none'
    sampling.corrector = 'mala'
    sampling.n_steps_each = 10
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 2 #0.075

    return config