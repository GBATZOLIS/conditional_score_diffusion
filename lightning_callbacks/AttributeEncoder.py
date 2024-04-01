import torch
import torchvision
from pytorch_lightning.callbacks import Callback
from . import utils
from math import sqrt

@utils.register_callback(name='AttributeEncoder')
class AttributeEncoderVisualizationCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = pl_module.current_epoch
        vis_freq = pl_module.config.training.visualisation_freq
        
        dataloader_iterator = iter(trainer.datamodule.val_dataloader())
        num_batches = 1
        for _ in range(num_batches):
            try:
                x, attributes = next(dataloader_iterator)
                x, attributes = x.to(pl_module.device), attributes.to(pl_module.device)
            except StopIteration:
                print('Requested number of batches exceeds the number of batches available in the val dataloader.')
                break
            
            if current_epoch % vis_freq == 0:
                # Generate conditional samples
                conditional_samples = pl_module.sample(attributes)
                accuracy_mean, accuracy_std = self.get_accuracy(trainer, pl_module, pl_module.sampling_eps, conditional_samples, attributes)
                if trainer.global_rank == 0 and pl_module.logger:
                    pl_module.logger.experiment.add_scalar('generated_acc', accuracy_mean, current_epoch) #accuracy mean
                    #pl_module.logger.experiment.add_scalar('generated_acc_std', accuracy_std, current_epoch) #accuracy std

                # Create a grid of original images
                num_rows = int(sqrt(x.size(0)))
                original_images_grid = torchvision.utils.make_grid(x, nrow=num_rows, normalize=True, scale_each=True)

                # Create a grid of conditional samples
                conditional_images_grid = torchvision.utils.make_grid(conditional_samples, nrow=num_rows, normalize=True, scale_each=True)

                # Concatenate the original and conditional samples grids
                concatenated_grid = torch.cat((original_images_grid, conditional_images_grid), 2)  # Concatenate side by side

                # Log the concatenated grid to TensorBoard
                tag = f'Original and Generated with same attributes'
                if trainer.global_rank == 0:
                    pl_module.logger.experiment.add_image(tag, concatenated_grid, current_epoch)

            accuracies = {}
            accuracy_stds = {}
            for time in [pl_module.sampling_eps, 0.25, 0.5, 0.75, pl_module.sde.T]:
                accuracy_mean, accuracy_std = self.get_accuracy(trainer, pl_module, time, x, attributes)
                accuracies[f'Time_{time:.2f}'] = accuracy_mean
                accuracy_stds[f'Time_{time:.2f}'] = accuracy_std

            # Log all accuracies together with the current epoch using add_scalars
            if trainer.global_rank == 0 and pl_module.logger:
                pl_module.logger.experiment.add_scalars('val_acc', accuracies, current_epoch)
                #pl_module.logger.experiment.add_scalars('val_acc_std', accuracy_stds, current_epoch)

    
    def get_accuracy(self, trainer, pl_module, t, x, attributes):
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        z = torch.randn_like(x)

        mean, std = pl_module.sde.marginal_prob(x, t_tensor)
        perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * z

        logits = pl_module.attribute_encoder(perturbed_x, t_tensor) #output size: (batchsize, num_features, num_classes)
        predictions = torch.argmax(logits, dim=-1) #choose as prediction the class with the highest logprobability

        correct_predictions = (predictions == attributes).float()
        accuracy_per_feature = correct_predictions.mean(dim=0) #calculate the accuracy for each feature
        
        mean_accuracy = accuracy_per_feature.mean()  # Calculate the average accuracy over all features
        std_accuracy = accuracy_per_feature.std()  # Calculate the standard deviation of accuracy over all features

        return mean_accuracy.item(), std_accuracy.item()

    def visualise_samples(self, trainer, pl_module, samples, time):
        # Log sampled images for a specific diffusion time
        sample_imgs = samples.cpu()
        grid_images = torchvision.utils.make_grid(sample_imgs, normalize=True, scale_each=True)
        tag = f'generated_images/Time_{time:.2f}'
        if trainer.global_rank == 0:
            pl_module.logger.experiment.add_image(tag, grid_images, pl_module.current_epoch)

@utils.register_callback(name='attribute_conditional')
class AttributeEncoderVisualizationCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = pl_module.current_epoch
        vis_freq = pl_module.config.training.visualisation_freq
        
        dataloader_iterator = iter(trainer.datamodule.val_dataloader())
        num_batches = 1
        for _ in range(num_batches):
            try:
                x, attributes = next(dataloader_iterator)
                x, attributes = x.to(pl_module.device), attributes.to(pl_module.device)
            except StopIteration:
                print('Requested number of batches exceeds the number of batches available in the val dataloader.')
                break
            
            if current_epoch % vis_freq == 0:
                # Generate conditional samples
                conditional_samples = pl_module.sample(attributes)
                # Create a grid of original images
                num_rows = int(sqrt(x.size(0)))
                original_images_grid = torchvision.utils.make_grid(x, nrow=num_rows, normalize=True, scale_each=True)
                # Create a grid of conditional samples
                conditional_images_grid = torchvision.utils.make_grid(conditional_samples, nrow=num_rows, normalize=True, scale_each=True)
                # Concatenate the original and conditional samples grids
                concatenated_grid = torch.cat((original_images_grid, conditional_images_grid), 2)  # Concatenate side by side
                # Log the concatenated grid to TensorBoard
                tag = f'Original and Generated with same attributes'
                if trainer.global_rank == 0:
                    pl_module.logger.experiment.add_image(tag, concatenated_grid, current_epoch)

@utils.register_callback(name='attribute_conditional_encoder')
class AttributeConditionalEncoderVisualizationCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = pl_module.current_epoch
        vis_freq = pl_module.config.training.visualisation_freq
        
        if current_epoch % vis_freq == 0:
            dataloader_iterator = iter(trainer.datamodule.val_dataloader())
            attribute_to_index_map = trainer.datamodule.get_attribute_to_index_map()
            attributes_to_flip = ['Eyeglasses']
            num_batches = 1
            for _ in range(num_batches):
                try:
                    x, y = next(dataloader_iterator)
                    x, y = x.to(pl_module.device), y.to(pl_module.device)
                except StopIteration:
                    print('Requested number of batches exceeds the number of batches available in the val dataloader.')
                    break
                
                #encode, flip, decode
                z, _ = pl_module.encode(x, y)
                flipped_y = pl_module.flip_attributes(y, attributes_to_flip, attribute_to_index_map)
                x_flipped = pl_module.decode(flipped_y, z)
                
                # Create a grid of original images
                num_rows = int(sqrt(x.size(0)))
                original_images_grid = torchvision.utils.make_grid(x, nrow=num_rows, normalize=True, scale_each=True)
                # Create a grid of conditional samples
                flipped_images_grid = torchvision.utils.make_grid(x_flipped, nrow=num_rows, normalize=True, scale_each=True)
                # Concatenate the original and conditional samples grids
                concatenated_grid = torch.cat((original_images_grid, flipped_images_grid), 2)  # Concatenate side by side
                # Log the concatenated grid to TensorBoard
                tag = f'Original and Flipped Attributes'
                if trainer.global_rank == 0:
                    pl_module.logger.experiment.add_image(tag, concatenated_grid, current_epoch)
