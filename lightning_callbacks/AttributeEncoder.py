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
        
        dataloader_iterator = iter(trainer.datamodule.val_dataloader())
        num_batches = 1
        for _ in range(num_batches):
            try:
                x, attributes = next(dataloader_iterator)
                x, attributes = x.to(pl_module.device), attributes.to(pl_module.device)
            except StopIteration:
                print('Requested number of batches exceeds the number of batches available in the val dataloader.')
                break
            
            if current_epoch % 5 == 0:
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

            accuracies = {}
            for time in [pl_module.sampling_eps, 0.25, 0.5, 0.75, pl_module.sde.T]:
                accuracy, perturbed_images = self.get_accuracy_and_perturbed_images(trainer, pl_module, time, x, attributes)
                accuracies[f'Time_{time:.2f}'] = accuracy

                # Visualize the perturbed images for this diffusion time
                self.visualise_samples(trainer, pl_module, perturbed_images, time)

            # Log all accuracies together with the current epoch using add_scalars
            if trainer.global_rank == 0 and pl_module.logger:
                pl_module.logger.experiment.add_scalars('val_accuracy', accuracies, current_epoch)


    def get_accuracy_and_perturbed_images(self, trainer, pl_module, t, x, attributes):
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        z = torch.randn_like(x)

        mean, std = pl_module.sde.marginal_prob(x, t_tensor)
        perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * z

        logits = pl_module.attribute_encoder(perturbed_x, t_tensor)
        predictions = torch.argmax(logits, dim=-1)

        correct_predictions = (predictions == attributes).float()
        accuracy_per_feature = correct_predictions.mean(dim=0)
        mean_accuracy = accuracy_per_feature.mean()

        return mean_accuracy.item(), perturbed_x

    def visualise_samples(self, trainer, pl_module, samples, time):
        # Log sampled images for a specific diffusion time
        sample_imgs = samples.cpu()
        grid_images = torchvision.utils.make_grid(sample_imgs, normalize=True, scale_each=True)
        tag = f'generated_images/Time_{time:.2f}'
        if trainer.global_rank == 0:
            pl_module.logger.experiment.add_image(tag, grid_images, pl_module.current_epoch)

