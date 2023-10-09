import os 
import torch
from pytorch_lightning.callbacks import Callback
from utils import scatter, plot, compute_grad, create_video, hist
import torchvision
from . import utils
from plot_utils import plot_curl, plot_vector_field, plot_spectrum, plot_norms, plot_distribution
import numpy as np
from models import utils as mutils
from pytorch_lightning.callbacks import ModelCheckpoint
import datetime
import pickle
from dim_reduction import get_manifold_dimension
import logging

@utils.register_callback(name='configuration')
class ConfigurationSetterCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        config = pl_module.config
        # Configure SDE
        pl_module.configure_sde(pl_module.config)
        
        # Configure trainining and validation loss functions.
        pl_module.train_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=True)
        pl_module.eval_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=False)

        # If log_path exists make sure you are resuming
        log_path = os.path.join(config.logging.log_path, config.logging.log_name)
        
        #this needs improvement. you might have created the log_path in an unsuccessful run but you might have nothing there (checkpoints etc.)
        #if config.model.checkpoint_path is None and os.path.exists(log_path):
        #    print('LOGGING PATH EXISTS BUT NOT RESUMING FROM CHECKPOINT!')
        #    raise RuntimeError('LOGGING PATH EXISTS BUT NOT RESUMING FROM CHECKPOINT!')

        # Pickle the config file 
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        with open(os.path.join(log_path, 'config.pkl'), 'wb') as file:
            pickle.dump(config, file)

        # Create a log file
        logging.basicConfig(handlers=[logging.FileHandler(filename="./log_records.txt", encoding='utf-8')], level=logging.DEBUG, force=True)
    
    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.configure_sde(pl_module.config)


@utils.register_callback(name='decreasing_variance_configuration')
class DecreasingVarianceConfigurationSetterCallback(ConfigurationSetterCallback):
    def __init__(self, config):
        super().__init__()
        self.sigma_max_y_fn = get_reduction_fn(y0=config.model.sigma_max_y, 
                                               xk=config.model.reach_target_steps, 
                                               yk=config.model.sigma_max_y_target)
        
        self.sigma_min_y_fn = get_reduction_fn(y0=config.model.sigma_min_y, 
                                               xk=config.model.reach_target_steps, 
                                               yk=config.model.sigma_min_y_target)


    def on_fit_start(self, trainer, pl_module):
        # Configure SDE
        pl_module.configure_sde(pl_module.config)
        
        # Configure trainining and validation loss functions.
        pl_module.train_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=True)
        pl_module.eval_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=False)

    def reconfigure_conditioning_sde(self, trainer, pl_module):
        #calculate current sigma_max_y and sigma_min_y
        current_sigma_max_y = self.sigma_max_y_fn(pl_module.global_step)
        current_sigma_min_y = self.sigma_min_y_fn(pl_module.global_step)

        # Reconfigure SDE
        pl_module.reconfigure_conditioning_sde(pl_module.config, current_sigma_min_y, current_sigma_max_y)
        
        # Reconfigure trainining and validation loss functions. -  we might not need to reconfigure the losses.
        pl_module.train_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=True)
        pl_module.eval_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=False)
        
        return current_sigma_min_y, current_sigma_max_y

    def on_sanity_check_start(self, trainer, pl_module):
        self.reconfigure_conditioning_sde(trainer, pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        current_sigma_min_y, current_sigma_max_y = self.reconfigure_conditioning_sde(trainer, pl_module)
        pl_module.sigma_max_y = torch.tensor(current_sigma_max_y).float()
        pl_module.sigma_min_y = torch.tensor(current_sigma_min_y).float()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        current_sigma_min_y, current_sigma_max_y = self.reconfigure_conditioning_sde(trainer, pl_module)
        
        pl_module.sigma_max_y = torch.tensor(current_sigma_max_y).float()
        pl_module.logger.experiment.add_scalar('sigma_max_y', current_sigma_max_y, pl_module.global_step)
        
        pl_module.sigma_min_y = torch.tensor(current_sigma_min_y).float()
        pl_module.logger.experiment.add_scalar('sigma_min_y', current_sigma_min_y, pl_module.global_step)

    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.configure_sde(config = pl_module.config, 
                                sigma_min_y = pl_module.sigma_min_y,
                                sigma_max_y = pl_module.sigma_max_y)


def get_reduction_fn(y0, xk, yk):
    #get the reduction function that starts at y0 and reaches point yk at xk steps.
    #the function follows an inverse multiplicative rate.
    def f(x):
        return xk*yk*y0/(x*(y0-yk)+xk*yk)
    return f

def get_deprecated_sigma_max_y_fn(reduction, reach_target_in_epochs, starting_transition_iterations):
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
                

@utils.register_callback(name='base')
class ImageVisualizationCallback(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.show_evolution = show_evolution

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = pl_module.current_epoch
        if (current_epoch+1) % 100 == 0:
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
        grid_images = torchvision.utils.make_grid(sample_imgs, normalize=True, scale_each=True)
        pl_module.logger.experiment.add_image('generated_images_%d' % pl_module.current_epoch, grid_images, pl_module.current_epoch)
    
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

@utils.register_callback(name='2DSamplesVisualization')
class TwoDimVizualizer(Callback):
    # SHOW EVOLUTION DOES NOT WORK AT THE MOMENT !
    def __init__(self, show_evolution=False):
        super().__init__()
        self.evolution = False #show_evolution

    def on_train_start(self, trainer, pl_module):
        samples, _ = pl_module.sample()
        self.visualise_samples(samples, pl_module)
        if pl_module.config.training.plot_ode:
            ode_samples, _ = pl_module.sample(ode=True)
            self.visualise_samples(ode_samples, pl_module, 'ode')
        
        if self.evolution:
             self.visualise_evolution(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch % 500 == 0:
            samples, _ = pl_module.sample()
            self.visualise_samples(samples, pl_module)
            if pl_module.config.training.plot_ode:
                ode_samples, _ = pl_module.sample(ode=True)
                self.visualise_samples(ode_samples, pl_module, 'ode')

        if self.evolution and pl_module.current_epoch % 2500 == 0 and pl_module.current_epoch != 0:
            self.visualise_evolution(pl_module)

    def visualise_samples(self, samples, pl_module, method='sde'):
        samples_np =  samples.cpu().numpy()
        image = scatter(samples_np[:,0],samples_np[:,1], 
                        title=f'{method}_samples epoch: ' + str(pl_module.current_epoch))
        pl_module.logger.experiment.add_image(f'samples_{method}', image, pl_module.current_epoch)           
        return image

    def visualise_evolution(self, pl_module):
        times=[0., .25, .5, .75, 1]
        images=[]
        for t in times:
            image=self.visualise_samples(pl_module, 'samples at time ' + str(t), t)
            images.append(image)
        grid = torchvision.utils.make_grid(images)
        pl_module.logger.experiment.add_image('samples evolution', grid, pl_module.current_epoch)
    
    def visualise_evolution_video(self, evolution, pl_module):
        title = 'samples epoch: ' + str(pl_module.current_epoch)
        video_tensor = create_video(evolution, 
                                    title=title,
                                    xlim=[-1,1],
                                    ylim=[-1,1])
        tag='Evolution_epoch_%d' % pl_module.current_epoch
        pl_module.logger.experiment.add_video(tag=tag, vid_tensor=video_tensor, fps=video_tensor.size(1)//20)

@utils.register_callback(name='2DCurlVisualization')
class CurlVizualizer(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.evolution = show_evolution

    def on_train_start(self, trainer, pl_module):
        self.visualise_curl(pl_module)
        if self.evolution:
            self.visualise_curl_evolution(pl_module)

    def on_validation_epoch_end(self,trainer, pl_module):
        if pl_module.current_epoch % 500 == 0:
            self.visualise_curl(pl_module)
        if self.evolution and pl_module.current_epoch % 2500 == 0:
            self.visualise_curl_evolution(pl_module)

    def visualise_curl(self, pl_module):
        image=plot_curl(pl_module, 'curl')
        pl_module.logger.experiment.add_image('curl', image, pl_module.current_epoch)
    
    def visualise_curl_evolution(self, pl_module):
        times=[0., .25, .5, .75, 1]
        images=[]
        for t in times:
            image=plot_curl(pl_module, 'curl at time ' + str(t), t)
            images.append(image)
        grid = torchvision.utils.make_grid(images)
        pl_module.logger.experiment.add_image('curl evolution', grid, pl_module.current_epoch)
        
        # video_tensor = torch.stack(images).unsqueeze(0)
        # tag='Curl_evolution_epoch_%d' % pl_module.current_epoch
        # pl_module.logger.experiment.add_video(tag=tag, vid_tensor=video_tensor, fps=video_tensor.size(1))
        


@utils.register_callback(name='2DVectorFieldVisualization')
class VectorFieldVizualizer(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.evolution = show_evolution

    def on_train_start(self, trainer, pl_module):
        self.visualise_vector_filed(pl_module)
        if self.evolution:
            self.visualise_vector_field_evolution(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch % 500 == 0:
            self.visualise_vector_filed(pl_module)
        if self.evolution and pl_module.current_epoch % 2500 == 0:
            self.visualise_vector_field_evolution(pl_module)

    def visualise_vector_filed(self, pl_module):
        image=plot_vector_field(pl_module, 'stream lines')
        pl_module.logger.experiment.add_image('stream lines', image, pl_module.current_epoch)
    
    def visualise_vector_field_evolution(self, pl_module):
        times=[0., .25, .5, .75, 1]
        images=[]
        for t in times:
            image=plot_vector_field(pl_module, 'stream lines at time ' + str(t), t)
            images.append(image)
        grid = torchvision.utils.make_grid(images)
        pl_module.logger.experiment.add_image('stream lines evolution', grid, pl_module.current_epoch)
        
        # video_tensor = torch.stack(images).unsqueeze(0)
        # tag='Stream_lines_evolution_epoch_%d' % pl_module.current_epoch
        # pl_module.logger.experiment.add_video(tag=tag, vid_tensor=video_tensor, fps=video_tensor.size(1))


@utils.register_callback(name='Conditional2DVisualization')
class ConditionalTwoDimVizualizer(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.evolution = show_evolution

    def on_train_start(self, trainer, pl_module):
        pass

    def on_validation_epoch_end(self,trainer, pl_module):
        batch_size=pl_module.config.validation.batch_size
        if pl_module.current_epoch % 500 == 0:
            ys = torch.tensor([0,.5,1,2]).to(pl_module.device)
            for y in ys:
                samples, _ = pl_module.sample(y.repeat(batch_size))
                self.visualise_samples(samples, y, pl_module)

    def visualise_samples(self, samples, y, pl_module):
        # log sampled images
        samples_np =  samples.cpu().numpy()
        image = scatter(samples_np[:,0],samples_np[:,1], 
                        title='samples epoch: ' + str(pl_module.current_epoch) + ' y = ' + str(y.item()))
        pl_module.logger.experiment.add_image('samples y = ' + str(y.item()), image, pl_module.current_epoch)
    def visualise_evolution(self, evolution, pl_module):
        pass


@utils.register_callback(name='Conditional1DVisualization')
class ConditionalTwoDimVizualizer(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.evolution = show_evolution

    def on_train_start(self, trainer, pl_module):
        pass

    def on_validation_epoch_end(self,trainer, pl_module):
        batch_size=pl_module.config.validation.batch_size
        if pl_module.current_epoch % 250 == 0:
            ys = torch.tensor([0,.5,1,2]).to(pl_module.device)
            for y in ys:
                samples, _ = pl_module.sample(y.repeat(batch_size))
                self.visualise_samples(samples, y, pl_module)

    def visualise_samples(self, samples, y, pl_module):
        # log sampled images
        image = hist(samples)
        pl_module.logger.experiment.add_image('samples y = ' + str(y.item()), image, pl_module.current_epoch)

    def visualise_evolution(self, evolution, pl_module):
        pass

@utils.register_callback(name='FisherDivergence')
class FisherDivergence(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if pl_module.current_epoch % 1 == 0:
            eps=1e-5
            t = torch.rand(batch.shape[0], device=batch.device) * (pl_module.sde.T - eps) + eps
            z = torch.randn_like(batch)
            mean, std = pl_module.sde.marginal_prob(batch, t)
            perturbed_data = mean + std[(...,) + (None,) * len(batch.shape[1:])] * z
            g2 = pl_module.sde.sde(torch.zeros_like(batch), t)[1] ** 2
            score_fn = mutils.get_score_fn(pl_module.sde, pl_module.score_model, train=False, continuous=True)
            model_score = score_fn(perturbed_data, t)
            gt_score = trainer.datamodule.data.ground_truth_score(perturbed_data, std)
            fisher_div = torch.mean(g2 * torch.linalg.norm(gt_score - model_score, dim=1)**2)
            pl_module.log('fisher_divergence', fisher_div, on_step=False, on_epoch=True, prog_bar=True, logger=True)


def sample_model_score(batch, pl_module):
    eps=1e-5
    t = torch.rand(batch.shape[0], device=batch.device) * (pl_module.sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = pl_module.sde.marginal_prob(batch, t)
    perturbed_data = mean + std[(...,) + (None,) * len(batch.shape[1:])] * z
    g2 = pl_module.sde.sde(torch.zeros_like(batch), t)[1] ** 2
    score_fn = mutils.get_score_fn(pl_module.sde, pl_module.score_model, train=False, continuous=True)
    return score_fn(perturbed_data, t)

@utils.register_callback(name='ScoreSpectrumVisualization')
class ScoreSpectrumVisualization(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.evolution = False #show_evolution  

    def on_validation_epoch_end(self, trainer, pl_module):
        config = pl_module.config
        if (pl_module.current_epoch+1) %  config.logging.svd_frequency == 0:
            config.model.checkpoint_path = os.path.join(config.logging.log_path, config.logging.log_name, "checkpoints/best/last.ckpt")
            name=f'svd_{pl_module.current_epoch}'
            try:
                if config.logging.save_svd:
                    get_manifold_dimension(config = config, name=name, return_svd=False)
                    path = os.path.join(config.logging.log_path, config.logging.log_name, 'svd', f'{name}.pkl')
                    with open(path, 'rb') as f:
                        svd = pickle.load(f)
                else:
                    svd = get_manifold_dimension(config = config, name=name, return_svd=True)
                image = plot_spectrum(svd, return_tensor=True, mode='all')
                image_distro, dims = plot_distribution(svd, return_tensor=True, mode='all')
                dim = np.mean(dims)

                pl_module.logger.experiment.add_image('score specturm', image, pl_module.current_epoch)
                pl_module.logger.experiment.add_image('dim_distribution', image_distro, pl_module.current_epoch)
                pl_module.log('dim', dim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                
            except Exception as e:
                logging.warning('Could not create a score spectrum')
                logging.error(e)

@utils.register_callback(name='KSphereEvaluation')
class KSphereEvaluation(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.evolution = False #show_evolution

    def on_validation_epoch_end(self,trainer, pl_module):
        config = pl_module.config
        if pl_module.current_epoch % config.logging.svd_frequency == 0:
            
            samples, _ = pl_module.sample(num_samples=1000)
            min_norm=torch.linalg.norm(samples, dim=1).min().item()
            max_norm=torch.linalg.norm(samples, dim=1).max().item()
            mean_norm=torch.linalg.norm(samples, dim=1).mean().item()
            image = plot_norms(samples=samples, return_tensor=True)
            pl_module.log('min_norm', min_norm, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            pl_module.log('max_norm', max_norm, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            pl_module.log('mean_norm', mean_norm, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            pl_module.logger.experiment.add_image('sample_norms_hist', image, pl_module.current_epoch)


# CHECKPOINTS
@utils.register_callback(name='CheckpointTopK')
class CheckpointTopK(ModelCheckpoint):
    def __init__(self, config):
        super().__init__(dirpath=os.path.join(config.logging.log_path, config.logging.log_name, 'checkpoints', 'best'),
                                        monitor='eval_loss_epoch',
                                        filename='{epoch}--{eval_loss_epoch:.3f}',
                                        save_last=True,
                                        save_top_k=config.logging.top_k,
                                        #train_time_interval=timedelta(hours=1)
                        )

@utils.register_callback(name='CheckpointEveryNepochs')
class CheckpointEveryNepochs(ModelCheckpoint):
    def __init__(self, config):
        super().__init__(dirpath=os.path.join(config.logging.log_path, config.logging.log_name, 'checkpoints', 'epochs'),
                                        monitor='eval_loss_epoch',
                                        filename='{epoch}',
                                        save_last=False,
                                        every_n_epochs=config.logging.every_n_epochs
                        )

@utils.register_callback(name='CheckpointTime')
class CheckpointTime(ModelCheckpoint):
    def __init__(self, config):
        super().__init__(dirpath=os.path.join(config.logging.log_path, config.logging.log_name, 'checkpoints', 'time'),
                                        monitor='eval_loss_epoch',
                                        filename=datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                                        save_last=False,
                                        train_time_interval=config.logging.envery_timedelta
                        )


