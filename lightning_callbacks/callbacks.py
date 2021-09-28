import torch
from pytorch_lightning.callbacks import Callback
from utils import scatter, plot, compute_grad, create_video
from models.ema import ExponentialMovingAverage
import torchvision
from . import utils
import numpy as np
from lightning_callbacks.HaarMultiScaleCallback import normalise_per_image

@utils.register_callback(name='configuration')
class ConfigurationSetterCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        # Configure SDE
        pl_module.configure_sde(pl_module.config)
        
        # Configure trainining and validation loss functions.
        pl_module.train_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=True)
        pl_module.eval_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=False)

        # Configure default sampling shape
        pl_module.configure_default_sampling_shape(pl_module.config)
    
    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.configure_sde(pl_module.config)


@utils.register_callback(name='decreasing_variance_configuration')
class DecreasingVarianceConfigurationSetterCallback(ConfigurationSetterCallback):
    def __init__(self, reduction, reach_target_in_epochs, starting_transition_iterations):
        super().__init__()
        self.reduction = reduction
        self.reach_target_in_epochs = reach_target_in_epochs
        self.starting_transition_iterations = starting_transition_iterations
        self.sigma_max_y_fn = get_sigma_max_y_fn(reduction, reach_target_in_epochs, starting_transition_iterations)

    def on_fit_start(self, trainer, pl_module):
        # Configure SDE
        pl_module.configure_sde(pl_module.config)
        
        # Configure trainining and validation loss functions.
        pl_module.train_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=True)
        pl_module.eval_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=False)

    def reconfigure_sigma_max_y(self, trainer, pl_module):
        current_epoch, global_step = pl_module.current_epoch, pl_module.global_step
        sigma_max_y_start = pl_module.config.model.sigma_max_x
        sigma_max_y_target = pl_module.config.model.sigma_max_y

        #calculate current sigma_max_y
        current_sigma_max_y = self.sigma_max_y_fn(global_step, current_epoch, sigma_max_y_start, sigma_max_y_target)

        # Reconfigure SDE
        pl_module.configure_sde(pl_module.config, current_sigma_max_y)
        # Reconfigure trainining and validation loss functions. -  we might not need to reconfigure the losses.
        pl_module.train_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=True)
        pl_module.eval_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=False)
        
        return current_sigma_max_y

    def on_sanity_check_start(self, trainer, pl_module):
        _ = self.reconfigure_sigma_max_y(trainer, pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        current_sigma_max_y = self.reconfigure_sigma_max_y(trainer, pl_module)
        pl_module.sigma_max_y = torch.tensor(current_sigma_max_y).float()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        current_sigma_max_y = self.reconfigure_sigma_max_y(trainer, pl_module)
        pl_module.sigma_max_y = torch.tensor(current_sigma_max_y).float()
        pl_module.logger.experiment.add_scalar('sigma_max_y', current_sigma_max_y, pl_module.global_step)

    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.configure_sde(pl_module.config, sigma_max_y = pl_module.sigma_max_y)


def get_sigma_max_y_fn(reduction, reach_target_in_epochs, starting_transition_iterations):
    if reduction == 'linear':
        def sigma_max_y(global_step, current_epoch, start_value, target_value):
            if current_epoch >= reach_target_in_epochs:
                current_sigma_max_y = target_value
            else:
                current_sigma_max_y = start_value - current_epoch/reach_target_in_epochs*(start_value - target_value)

            return current_sigma_max_y
                
    elif reduction == 'inverse_exponentional':
        def sigma_max_y(global_step, current_epoch, start_value, target_value):
            x_prev = 0
            x_next = starting_transition_iterations
            x_add = starting_transition_iterations

            while global_step > x_next:
                x_add *= 2
                x_prev = x_next
                x_next = x_add + x_prev
                start_value = start_value/2

            target_value = start_value/2
            current_sigma_max_y = start_value - (global_step-x_prev)/(x_next-x_prev)*(start_value - target_value)
            return current_sigma_max_y
    else:
        raise NotImplementedError('Reduction type %s is not supported yet.' % reduction)

    return sigma_max_y
                

@utils.register_callback(name='ema')
class EMACallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        ema_rate = pl_module.config.model.ema_rate
        pl_module.ema = ExponentialMovingAverage(pl_module.parameters(), decay=ema_rate)

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        pl_module.ema.update(pl_module.parameters())

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.ema.store(pl_module.parameters())
        pl_module.ema.copy_to(pl_module.parameters())

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.ema.restore(pl_module.parameters())

@utils.register_callback(name='base')
class ImageVisualizationCallback(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.show_evolution = show_evolution

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch >= 0:
            if self.show_evolution:
                samples, sampling_info = pl_module.sample(show_evolution=True)
                evolution = sampling_info['evolution']
                self.visualise_evolution(evolution, pl_module)
            else:
                samples, _ = pl_module.sample(show_evolution=False)

            self.visualise_samples(samples, pl_module)

    def visualise_samples(self, samples, pl_module):
        # log sampled images
        sample_imgs =  samples.cpu()
        grid_images = torchvision.utils.make_grid(normalise_per_image(sample_imgs), normalize=False)
        pl_module.logger.experiment.add_image('generated_images', grid_images, pl_module.current_epoch)
    
    def visualise_evolution(self, evolution, pl_module):
        #to be implemented - has already been implemented for the conditional case
        return



@utils.register_callback(name='GradientVisualization')
class GradientVisualizer(Callback):

    def on_validation_epoch_end(self,trainer, pl_module):
        if pl_module.current_epoch % 500 == 0:
            _, sampling_info = pl_module.sample(show_evolution=True)
            evolution, times = sampling_info['evolution'], sampling_info['times']
            self.visualise_grad_norm(evolution, times, pl_module)

    def visualise_grad_norm(self, evolution, times, pl_module):
        grad_norm_t =[]
        for i in range(evolution.shape[0]):
            t = times[i]
            samples = evolution[i]
            vec_t = torch.ones(times.shape[0], device=t.device) * t
            gradients = compute_grad(f=pl_module.score_model, x=samples, t=vec_t)
            grad_norm = gradients.norm(2, dim=1).max().item()
            grad_norm_t.append(grad_norm)
        image = plot(times.cpu().numpy(),
                        grad_norm_t,
                        'Gradient Norms Epoch: ' + str(pl_module.current_epoch)
                        )
        pl_module.logger.experiment.add_image('grad_norms', image, pl_module.current_epoch)

@utils.register_callback(name='2DVisualization')
class TwoDimVizualizer(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.evolution = show_evolution

    def on_train_start(self, trainer, pl_module):
        # pl_module.logxger.log_hyperparams(params=pl_module.config.to_dict())
        samples, _ = pl_module.sample()
        self.visualise_samples(samples, pl_module)

    def on_validation_epoch_end(self,trainer, pl_module):
        if pl_module.current_epoch % 500 == 0 \
            and pl_module.current_epoch % 2500 != 0:
            samples, _ = pl_module.sample()
            self.visualise_samples(samples, pl_module)
        if self.evolution and pl_module.current_epoch % 2500 == 0:
            samples, sampling_info = pl_module.sample(show_evolution=True)
            evolution = sampling_info['evolution']
            self.visualise_evolution(evolution, pl_module)

    def visualise_samples(self, samples, pl_module):
        # log sampled images
        samples_np =  samples.cpu().numpy()
        image = scatter(samples_np[:,0],samples_np[:,1], 
                        title='samples epoch: ' + str(pl_module.current_epoch))
        pl_module.logger.experiment.add_image('samples', image, pl_module.current_epoch)
    
    def visualise_evolution(self, evolution, pl_module):
        title = 'samples epoch: ' + str(pl_module.current_epoch)
        video_tensor = create_video(evolution, 
                                    title=title,
                                    xlim=[-1,1],
                                    ylim=[-1,1])
        tag='Evolution_epoch_%d' % pl_module.current_epoch
        pl_module.logger.experiment.add_video(tag=tag, vid_tensor=video_tensor, fps=video_tensor.size(1)//20)


