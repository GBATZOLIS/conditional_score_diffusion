import os 
import torch
from pytorch_lightning.callbacks import Callback
from utils import scatter, plot, compute_grad, create_video, hist
from models.ema import ExponentialMovingAverage
import torchvision
from . import utils
from plot_utils import plot_curl, plot_vector_field, plot_spectrum, plot_norms
import numpy as np
from models import utils as mutils
from pytorch_lightning.callbacks import ModelCheckpoint
import datetime
import pickle
from dim_reduction import get_manifold_dimension
import logging
from models import utils as mutils
from configs.utils import fix_config
import copy

from torch.distributions import Distribution
from scipy.interpolate import PchipInterpolator
import io
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

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
                

'''
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
'''

@utils.register_callback(name='load_new_prior')
class LoadPriorCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config=config
    
    def on_train_start(self, trainer, pl_module):
        pl_module.unconditional_score_model = mutils.load_prior_model(self.config)
        pl_module.unconditional_score_model = pl_module.unconditional_score_model.to(pl_module.device)
        pl_module.unconditional_score_model.freeze()


@utils.register_callback(name='base')
class ImageVisualizationCallback(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.show_evolution = show_evolution

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = pl_module.current_epoch
        freq = pl_module.config.training.visualisation_freq
        if (current_epoch+1) % freq == 0:
            if self.show_evolution:
                samples, sampling_info = pl_module.sample(show_evolution=True)
                evolution = sampling_info['evolution']
                self.visualise_evolution(evolution, pl_module)
            else:
                samples, _ = pl_module.sample(show_evolution=False, p_steps=250)

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
        if self.evolution:
             self.visualise_evolution(pl_module)

    def on_validation_epoch_end(self,trainer, pl_module):
        if pl_module.current_epoch % 500 == 0:
            samples, _ = pl_module.sample()
            self.visualise_samples(samples, pl_module)
        if self.evolution and pl_module.current_epoch % 2500 == 0 and pl_module.current_epoch != 0:
            self.visualise_evolution(pl_module)

    def visualise_samples(self, samples, pl_module):
        samples_np =  samples.cpu().numpy()
        image = scatter(samples_np[:,0],samples_np[:,1], 
                        title='samples epoch: ' + str(pl_module.current_epoch))
        pl_module.logger.experiment.add_image('samples', image, pl_module.current_epoch)
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

@utils.register_callback(name='ScoreSpecturmVisualization')
class ScoreSpecturmVisualization(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.evolution = False #show_evolution

        

    def on_validation_epoch_end(self, trainer, pl_module):
        config = pl_module.config
        if pl_module.current_epoch %  config.logging.svd_frequency == 0:
            config.model.checkpoint_path = os.path.join(config.logging.log_path, config.logging.log_name, "checkpoints/best/last.ckpt")
            name=f'svd_{pl_module.current_epoch}'
            try:
                get_manifold_dimension(config = config, name=name)
                path = os.path.join(config.logging.log_path, config.logging.log_name, 'svd', f'{name}.pkl')
                with open(path, 'rb') as f:
                    svd = pickle.load(f)
                singular_values = svd['singular_values']
                image = plot_spectrum(singular_values=singular_values, return_tensor=True)
                pl_module.logger.experiment.add_image('score specturm', image, pl_module.current_epoch)
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


@utils.register_callback(name='celeba_distribution_shift')
class DistributionShift(Callback):

    def __init__(self, config) -> None:
        super().__init__()

        # create config for celebA
        celeb_config = copy.deepcopy(fix_config(config)) # do I need to copy here?
        celeb_config.data.dataset = 'celebA-HQ-160'
        celeb_config.data.attributes = ['Male']
        celeb_config.data.normalization_mode = 'gd'
        self.celeb_config = celeb_config

        # load data
        from lightning_data_modules.ImageDatasets import CelebAAnnotatedDataset
        from torch.utils.data import DataLoader
        celeb_dataset = CelebAAnnotatedDataset(celeb_config, phase='val') #test and val are the same
        test_dataloader = DataLoader(celeb_dataset, batch_size=celeb_config.validation.batch_size, shuffle=False)
        self.celeb_batch = next(iter(test_dataloader))[0]

    def on_validation_epoch_start(self,trainer, pl_module):
        self.celeb_batch = self.celeb_batch.to(pl_module.device)
        if (pl_module.current_epoch+1) % pl_module.config.training.visualisation_freq == 0:
            reconstruction = pl_module.encode_n_decode(self.celeb_batch, p_steps=250,
                                                         use_pretrained=self.celeb_config.training.use_pretrained,
                                                         encoder_only=self.celeb_config.training.encoder_only,
                                                         t_dependent=self.celeb_config.training.t_dependent)
            
            reconstruction =  reconstruction.cpu()
            grid_reconstruction = torchvision.utils.make_grid(reconstruction, nrow=int(np.sqrt(self.celeb_batch.size(0))), normalize=True, scale_each=True)
            pl_module.logger.experiment.add_image('celeba_reconstruction', grid_reconstruction, pl_module.current_epoch)
            
            self.celeb_batch = self.celeb_batch.cpu()
            grid_batch = torchvision.utils.make_grid(self.celeb_batch, nrow=int(np.sqrt(self.celeb_batch.size(0))), normalize=True, scale_each=True)
            pl_module.logger.experiment.add_image('celeba_real', grid_batch)

            difference = torch.flatten(reconstruction, start_dim=1)-torch.flatten(self.celeb_batch, start_dim=1)
            L2norm = torch.linalg.vector_norm(difference, ord=2, dim=1)
            avg_L2norm = torch.mean(L2norm)
            pl_module.log('celeba_reconstruction_loss', avg_L2norm, logger=True)

@utils.register_callback(name='jan_georgios')
class JanGeorgios(Callback):
    def __init__(self, config) -> None:
        im_size = config.data.image_size
        super().__init__()
        import torchvision
        path_jan = 'images_for_manipulation/jan.jpg'
        jan=torchvision.io.read_image(path_jan)
        min_dim = min(jan.shape[1], jan.shape[2])
        offset = 0
        jan = jan[:, offset:(min_dim+offset), :min_dim]
        georgios = torchvision.io.read_image('images_for_manipulation/georgios.jpg')[:,25:225,50:250]
        resize = torchvision.transforms.Resize((im_size,im_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
        jan=resize(jan)
        georgios=resize(georgios)
        # # normalize to 0, 1
        # jan = jan/255
        # georgios=resize(georgios)
        # georgios = georgios/255
        # normalize to -1, 1
        self.jan = jan / 127.5 - 1
        self.georgios = georgios / 127.5 - 1

    def on_validation_epoch_start(self,trainer, pl_module):
        batch = torch.stack([self.jan, self.georgios]).to(pl_module.device)
        if (pl_module.current_epoch+1) % pl_module.config.training.visualisation_freq == 0:
            reconstruction = pl_module.encode_n_decode(batch, p_steps=250,
                                                         use_pretrained=pl_module.config.training.use_pretrained,
                                                         encoder_only=pl_module.config.training.encoder_only,
                                                         t_dependent=pl_module.config.training.t_dependent)
            
            reconstruction =  reconstruction.cpu()
            grid_reconstruction = torchvision.utils.make_grid(reconstruction, nrow=int(np.sqrt(batch.size(0))), normalize=True, scale_each=True)
            pl_module.logger.experiment.add_image('jan_georgios_reconstruction', grid_reconstruction, pl_module.current_epoch)
            
            batch = batch.cpu()
            grid_batch = torchvision.utils.make_grid(batch, nrow=int(np.sqrt(batch.size(0))), normalize=True, scale_each=True)
            pl_module.logger.experiment.add_image('jan_georgios_real', grid_batch)

            difference = torch.flatten(reconstruction, start_dim=1)-torch.flatten(batch, start_dim=1)
            L2norm = torch.linalg.vector_norm(difference, ord=2, dim=1)
            avg_L2norm = torch.mean(L2norm)
            pl_module.log('jan_georgios_reconstruction_loss', avg_L2norm, logger=True)

@utils.register_callback(name='encoder_contribution')
class EncoderContribution(Callback):
    def __init__(self, config) -> None:
        super().__init__()

    def get_distribution_class(self, ):
        class SamplingDistribution(Distribution):
            def __init__(self, times, vals, device="cpu"):
                super().__init__()

                self.device = torch.device(device)
                self.times = torch.tensor(times, dtype=torch.float32, device=self.device)
                self.vals = torch.tensor(vals, dtype=torch.float32, device=self.device)
                self.inverse_cdf, self.cdf, self.density = self.get_interpolations()

            def get_interpolations(self):
                times = self.times.cpu().numpy()
                vals = self.vals.cpu().numpy()
                vals[0] = 0.
                cdf_y = np.cumsum(vals) 
                cdf_y = cdf_y / cdf_y.max()
                inverse_cdf = PchipInterpolator(cdf_y, times)
                cdf = PchipInterpolator(times, cdf_y)
                density = cdf.derivative(nu=1)
                return inverse_cdf, cdf, density
            
            def sample(self, shape):
                u = torch.rand(size=shape, device=self.device)
                sample = torch.from_numpy(self.inverse_cdf(u.cpu().numpy())).float().to(self.device)
                return sample

            def prob(self, val):
                return torch.from_numpy(self.density(val.cpu().numpy())).float().to(self.device)

            def log_prob(self, val):
                return torch.log(self.prob(val))

            def rsample(self, shape):
                return self.sample(shape)
        
        return SamplingDistribution

    def get_latent_correction_fn(self, encoder):
        def get_log_density_fn(encoder):
            def log_density_fn(x, z, t):
                latent_distribution_parameters = encoder(x, t)
                latent_dim = latent_distribution_parameters.size(1)//2
                mean_z = latent_distribution_parameters[:, :latent_dim]
                log_var_z = latent_distribution_parameters[:, latent_dim:]

                #print(x.size())
                print(z.size())
                print(mean_z.size())
                print(log_var_z.size())

                logdensity = -1/2*torch.sum(torch.square(z - mean_z)/log_var_z.exp(), dim=1)
                return logdensity
                
            return log_density_fn

        def latent_correction_fn(x, z, t):
            torch.set_grad_enabled(True)
            log_density_fn = get_log_density_fn(encoder)
            device = x.device
            x.requires_grad=True
            ftx = log_density_fn(x, z, t)
            grad_log_density = torch.autograd.grad(outputs=ftx, inputs=x,
                                            grad_outputs=torch.ones(ftx.size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            assert grad_log_density.size() == x.size()
            torch.set_grad_enabled(False)
            return grad_log_density

        return latent_correction_fn

    def on_validation_epoch_start(self, trainer, pl_module):
        sde = pl_module.sde
        encoder = pl_module.encoder
        unconditional_score_model = pl_module.unconditional_score_model
        device = pl_module.device
        eps = sde.sampling_eps
        T = sde.T

        if (pl_module.current_epoch+1) % pl_module.config.training.importance_freq == 0:
            dataloader = trainer.datamodule.val_dataloader()
            x = next(iter(dataloader))
            x = pl_module._handle_batch(x)
            x = x.to(device)
            z = pl_module.encode(x)

            encoder_correction_fn = self.get_latent_correction_fn(encoder)
            unconditional_score_fn = mutils.get_score_fn(sde,unconditional_score_model, 
                                                         conditional=False, train=False, continuous=True)  

            ts = torch.linspace(start=eps, end=T, steps=25)
            
            relative_contribution = [] 
            for t_ in ts:
                t = torch.ones(x.shape[0]).type_as(x)*t_

                z = torch.randn_like(x)
                mean, std = sde.marginal_prob(x, t)
                perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * z

                unconditional_score = unconditional_score_fn(perturbed_x, t)
                encoder_correction = encoder_correction_fn(perturbed_x, z, t)
                #total_score = unconditional_score + encoder_correction #Bayes rule
                
                unconditional_score_norm = torch.linalg.norm(unconditional_score.reshape(unconditional_score.shape[0], -1), dim=1)
                encoder_correction_norm = torch.linalg.norm(encoder_correction.reshape(encoder_correction.shape[0], -1), dim=1)
                
                ratio = encoder_correction_norm / unconditional_score_norm
                mean_ratio = torch.mean(ratio)
                relative_contribution.append(mean_ratio.item())
            
            times = ts.numpy()
            contributions = np.array(relative_contribution)

            t_dist = self.get_distribution_class()(times, contributions, device)
            pl_module.t_dist = t_dist

            #create the figure
            x = np.linspace(eps, 1, 1000)
            fig, ax = plt.subplots()
            ax.set_xlabel('x')
            ax.set_ylabel('probability density')
            ax.plot(x, pl_module.t_dist.density(x), label='density')
            samples = pl_module.t_dist.sample((10000,)).numpy()
            ax.hist(samples, bins='auto', density=True, range=(x.min(),x.max()), alpha=0.5, label='Histogram')
            ax.legend(loc="upper right")
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            image = Image.open(buf)
            image = ToTensor()(image)  # Convert the image to PyTorch tensor
            buf.close()
            
            # Log the image to TensorBoard
            trainer.logger.experiment.add_image('Distribution', image, global_step=pl_module.current_epoch)









