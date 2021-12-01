from . import utils
import torch
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid, save_image
import numpy as np
import lpips
from . import evaluation_tools as eval_tools
from pathlib import Path
import os
import matplotlib.pyplot as plt

def normalise(c, value_range=None):
    x = c.clone()
    if value_range is None:
        x -= x.min()
        x /= x.max()
    else:
        x -= value_range[0]
        x /= value_range[1]
    return x

def normalise_per_image(x, value_range=None):
    for i in range(x.size(0)):
        x[i,::] = normalise(x[i,::], value_range=value_range)
    return x

def normalise_evolution(evolution):
    normalised_evolution = torch.ones_like(evolution)
    for i in range(evolution.size(0)):
        normalised_evolution[i] = normalise_per_image(evolution[i])
    return normalised_evolution

def create_video_grid(evolution):
    video_grid = []
    for i in range(evolution.size(0)):
        video_grid.append(make_grid(evolution[i], nrow=int(np.sqrt(evolution[i].size(0))), normalize=False))
    return torch.stack(video_grid)


@utils.register_callback(name='paired')
class PairedVisualizationCallback(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.show_evolution = show_evolution

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = pl_module.current_epoch
        if current_epoch == 0 or current_epoch % 20 != 0:
            return 
        
        dataloader_iterator = iter(trainer.datamodule.val_dataloader())
        num_batches = 1
        for i in range(num_batches):
            try:
                y, x = next(dataloader_iterator)
            except StopIteration:
                print('Requested number of batches exceeds the number of batches available in the val dataloader.')
                break

            if self.show_evolution:
                print('sde_y sigma_max: %.5f ' % pl_module.sde['y'].sigma_max)
                conditional_samples, sampling_info = pl_module.sample(y.to(pl_module.device), show_evolution=True)
                evolution = sampling_info['evolution']
                self.visualise_evolution(evolution, pl_module, tag='val_joint_evolution_batch_%d_epoch_%d' % (i, current_epoch))
            else:
                conditional_samples, _ = pl_module.sample(y.to(pl_module.device), show_evolution=False)

            self.visualise_paired_samples(y, conditional_samples, x, pl_module, i+1)

    def visualise_paired_samples(self, y, x, gt, pl_module, batch_idx, phase='train'):
        # log sampled images
        y_norm, x_norm, gt_norm = normalise_per_image(y).cpu(), normalise_per_image(x).cpu(), normalise_per_image(gt).cpu()
        
        if y_norm.size(1) == 1 and y_norm.size(1) < gt.size(1): #colorization
            y_norm = y_norm.repeat(1, 3, 1, 1)

        concat_sample = torch.cat([y_norm, x_norm, gt_norm], dim=-1)
        grid_images = make_grid(concat_sample, nrow=int(np.sqrt(concat_sample.size(0))), normalize=False)
        pl_module.logger.experiment.add_image('generated_images_%sbatch_%d' % (phase, batch_idx), grid_images, pl_module.current_epoch)
    
    def visualise_evolution(self, evolution, pl_module, tag):
        norm_evolution_x = normalise_evolution(evolution['x'])
        norm_evolution_y = normalise_evolution(evolution['y'])
        joint_evolution = torch.cat([norm_evolution_y, norm_evolution_x], dim=-1)
        video_grid = create_video_grid(joint_evolution)
        pl_module.logger.experiment.add_video(tag, video_grid.unsqueeze(0), fps=50)

@utils.register_callback(name='test_paired')
class TestPairedVisualizationCallback(PairedVisualizationCallback):
    def __init__(self, show_evolution, eval_config, data_config, approach):
        super().__init__(show_evolution)
        #settings related to the conditional sampling function.
        self.predictor = eval_config.predictor
        self.corrector = eval_config.corrector
        self.p_steps = eval_config.p_steps
        self.c_steps = eval_config.c_steps
        self.denoise = eval_config.denoise
        self.use_path = eval_config.use_path
        
        #settings for determining the sampling process and saving the samples
        self.save_samples = eval_config.save_samples

        if self.save_samples:
            self.base_dir = eval_config.base_log_dir
            self.dataset = data_config.dataset
            self.task = data_config.task
            self.samples_dir = os.path.join(self.base_dir, self.task, self.dataset, approach, 'images', 'samples')
            self.gt_x_dir = os.path.join(self.base_dir, self.task, self.dataset, approach, 'images', 'x_gt')
            self.gt_y_dir = os.path.join(self.base_dir, self.task, self.dataset, approach, 'images','y_gt')
            
            Path(self.samples_dir).mkdir(parents=True, exist_ok=True)
            Path(self.gt_x_dir).mkdir(parents=True, exist_ok=True)
            Path(self.gt_y_dir).mkdir(parents=True, exist_ok=True)
        
        self.num_draws = eval_config.num_draws
        self.evaluation_metrics = eval_config.evaluation_metrics
        
        if not isinstance(eval_config.snr, list):
            self.snr = [eval_config.snr]
        else:
            self.snr = eval_config.snr #list of snr values to be tested.

        self.results = {}
        for e_snr in self.snr:
            #create the save directories
            if self.save_samples:
                e_snr_path = os.path.join(self.samples_dir, 'snr_%.3f' % e_snr)
                Path(e_snr_path).mkdir(parents=True, exist_ok=True)
                for draw in range(self.num_draws):
                    draw_e_snr_path = os.path.join(self.samples_dir, 'snr_%.3f' % e_snr, 'draw_%d' % (draw+1))
                    Path(draw_e_snr_path).mkdir(parents=True, exist_ok=True)

            self.results[e_snr] = {}
            for eval_metric in self.evaluation_metrics:
                self.results[e_snr][eval_metric]=[]

        #auxiliary counters and limits
        self.images_tested = 0
        self.test_batch_limit = eval_config.test_batch_limit

    def on_test_start(self, trainer, pl_module):
        pl_module.loss_fn_alex = lpips.LPIPS(net='alex').to(pl_module.device)

    def generate_metric_vals(self, y, x, pl_module, snr):
        metric_vals = {}
        for eval_metric in self.evaluation_metrics:
            metric_vals[eval_metric]=[]

        for draw in range(self.num_draws):
            #sample x conditioned on y
            samples, _ = pl_module.sample(y, show_evolution=False, 
                                          predictor=self.predictor, corrector=self.corrector, 
                                          p_steps=self.p_steps, c_steps=self.c_steps, snr=snr, 
                                          denoise=self.denoise, use_path=self.use_path) 
                    
            #some reverse diffused values might be slightly off - correct them. Bear in mind we podel p_epsilon not p_0...
            samples = torch.clamp(samples, min=0, max=1)

            #save the generated samples if self.save_samples is True
            if self.save_samples:
                samples_save_dir = os.path.join(self.samples_dir, 'snr_%.3f' % snr, 'draw_%d' % (draw+1))
                for i in range(samples.size(0)):
                    fp = os.path.join(samples_save_dir, '%d.png' % (self.images_tested+i+1))
                    save_image(samples[i, :, :, :], fp)

            if 'lpips' in self.evaluation_metrics:
                lpips_val = pl_module.loss_fn_alex(2*x-1, 2*samples-1).cpu()
                print('lpips_val.squeeze().size(): ', lpips_val.squeeze().size())
                metric_vals['lpips'].append(torch.mean(lpips_val.squeeze()).item())
                    
            #convert the torch tensors to numpy arrays for the remaining metric calculations
            numpy_samples = torch.swapaxes(samples.clone().cpu(), axis0=1, axis1=-1).numpy()*255
            numpy_gt = torch.swapaxes(x.clone().cpu(), axis0=1, axis1=-1).numpy()*255

            if 'psnr' in self.evaluation_metrics:
                metric_vals['psnr'] = eval_tools.calculate_mean_psnr(numpy_samples, numpy_gt)
                    
            if 'ssim' in self.evaluation_metrics:
                metric_vals['ssim'].append(eval_tools.calculate_mean_ssim(numpy_samples, numpy_gt))
                    
            if 'consistency' in self.evaluation_metrics:
                lr_synthetic = torch.swapaxes(eval_tools.resize(samples.cpu(), 1/pl_module.config.data.scale), axis0=1, axis1=-1).numpy()*255
                lr_gt = torch.swapaxes(eval_tools.resize(x.cpu(), 1/pl_module.config.data.scale), axis0=1, axis1=-1).numpy()*255
                print(lr_synthetic.shape)
                print(lr_gt.shape)
                metric_vals['consistency'].append(eval_tools.calculate_mean_psnr(lr_synthetic, lr_gt))
                    
            if 'diversity' in self.evaluation_metrics:
                samples=samples*255.
                metric_vals['diversity'].append(samples.to('cpu'))
        
        return metric_vals

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if batch_idx >= self.test_batch_limit:
            return 

        y, x = batch

        if self.save_samples:
            for i in range(x.size(0)):
                save_image(x[i, :, :, :], fp = os.path.join(self.gt_x_dir, '%d.png' % (self.images_tested+i+1)))
                save_image(y[i, :, :, :], fp = os.path.join(self.gt_y_dir, '%d.png' % (self.images_tested+i+1)))

        for e_snr in self.snr:
            metric_vals = self.generate_metric_vals(y, x, pl_module, e_snr)
                
            for eval_metric in self.evaluation_metrics:
                if eval_metric == 'diversity':
                    self.results[e_snr][eval_metric].append(float(torch.cat(metric_vals['diversity'], 0).std([0]).mean().cpu()))
                else:
                    self.results[e_snr][eval_metric].append(np.mean(metric_vals[eval_metric]))
        
        self.images_tested += x.size(0)
    
    def on_test_epoch_end(self, trainer, pl_module):
        for eval_metric in self.evaluation_metrics:
            fig = plt.figure()
            plt.title('%s' % eval_metric)
            mean_vals, snrs = [], []
            for e_snr in self.snr:
                mean_vals.append(np.mean(self.results[e_snr][eval_metric]))
                snrs.append(e_snr)
                #pl_module.logger.experiment.add_scalar(eval_metric, np.mean(self.results[e_snr][eval_metric]), e_snr)
            
            plt.scatter(snrs, mean_vals)
            plt.xlabel('snr')
            plt.ylabel('%s' % eval_metric)

            pl_module.logger.experiment.add_figure(eval_metric, fig)


@utils.register_callback(name='paired3D')
class PairedVisualizationCallback(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.show_evolution = show_evolution

    def convert_to_3D(self, x):
        if len(x.shape[1:]) == 3:
            x = torch.swapaxes(x, 1, -1).unsqueeze(1)
            print(x.size())
            return x
        elif len(x.shape[1:]) == 4:
            return x
        else:
            raise NotImplementedError('x dimensionality is not supported.')

    def generate_paired_video(self, pl_module, Y, I, cond_samples, dim, batch_idx):
        #dim: the sliced dimension (choices: 1,2,3)
        B = Y.size(0)

        if cond_samples is not None:
            raw_length = 1+cond_samples.size(0)+1
        else:
            raw_length = 2

        frames = Y.size(dim+1)
        video_grid = []
        for frame in range(frames):
            if dim==1:
                dim_cut = torch.zeros(tuple([B*raw_length, 1, I.shape[3], I.shape[4]])).type_as(Y)
            elif dim==2:
                dim_cut = torch.zeros(tuple([B*raw_length, 1, I.shape[2], I.shape[4]])).type_as(Y)
            elif dim==3:
                dim_cut = torch.zeros(tuple([B*raw_length, 1, I.shape[2], I.shape[3]])).type_as(Y)

            for i in range(B):
                if dim==1:
                    dim_cut[i*raw_length] = normalise(Y[i, 0, frame, :, :]).unsqueeze(0)
                    dim_cut[(i+1)*raw_length-1] = normalise(I[i, 0, frame, :, :]).unsqueeze(0)
                    if cond_samples is not None:
                        for j in range(cond_samples.size(0)):
                            dim_cut[i*raw_length+j+1] = normalise(cond_samples[j, i, 0, frame, :, :]).unsqueeze(0)
                elif dim==2:
                    dim_cut[i*raw_length] = normalise(Y[i, 0, :, frame, :]).unsqueeze(0)
                    dim_cut[(i+1)*raw_length-1] = normalise(I[i, 0, :, frame, :]).unsqueeze(0)
                    if cond_samples is not None:
                        for j in range(cond_samples.size(0)):
                            dim_cut[i*raw_length+j+1] = normalise(cond_samples[j, i, 0, :, frame, :]).unsqueeze(0)
                elif dim==3:
                    dim_cut[i*raw_length] = normalise(Y[i, 0, :, :, frame]).unsqueeze(0)
                    dim_cut[(i+1)*raw_length-1] = normalise(I[i, 0, :, :, frame]).unsqueeze(0)
                    if cond_samples is not None:
                        for j in range(cond_samples.size(0)):
                            dim_cut[i*raw_length+j+1] = normalise(cond_samples[j, i, 0, :, :, frame]).unsqueeze(0)

            grid_cut = make_grid(tensor=dim_cut, nrow=raw_length, normalize=False)
            video_grid.append(grid_cut)

        video_grid = torch.stack(video_grid, dim=0).unsqueeze(0)
        #print(video_grid.size())

        str_title = 'paired_video_epoch_%d_batch_%d_dim_%d' % (pl_module.current_epoch, batch_idx, dim)
        pl_module.logger.experiment.add_video(str_title, video_grid, pl_module.current_epoch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        current_epoch = pl_module.current_epoch
        if batch_idx!=2 or current_epoch == 0 or current_epoch % 50 != 0:
            return
        
        y, x = batch        
        cond_samples, _ = pl_module.sample(y.to(pl_module.device), show_evolution=self.show_evolution)
        val_rec_loss = torch.mean(torch.abs(x.to(pl_module.device)-cond_samples))
        pl_module.logger.experiment.add_scalar('val_rec_loss_epoch_%d_batch_%d' % (current_epoch, batch_idx), val_rec_loss)

        x = self.convert_to_3D(x).cpu()
        cond_samples = self.convert_to_3D(cond_samples).unsqueeze(0).cpu()
        y = self.convert_to_3D(y).cpu()
        

        for dim in [1, 2, 3]:
            self.generate_paired_video(pl_module, y, x, cond_samples, dim, batch_idx)