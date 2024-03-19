import torch
import torchvision
from math import sqrt

def check_increasing_classifier_guidance_strength(pl_module, datamodule, logger):
    pl_module.configure_sde(pl_module.config)
    dataloader_iterator = iter(datamodule.val_dataloader())
    x, attributes = next(dataloader_iterator)
    x, attributes = x.to(pl_module.device), attributes.to(pl_module.device)

    num_rows = int(sqrt(x.size(0)))
    original_images_grid = torchvision.utils.make_grid(x, nrow=num_rows, normalize=True, scale_each=True)
    for gamma in [2, 2.5]:
        conditional_samples = pl_module.sample(attributes, gamma)
        num_rows = int(sqrt(x.size(0)))
        
        # Create a grid of conditional samples
        conditional_images_grid = torchvision.utils.make_grid(conditional_samples, nrow=num_rows, normalize=True, scale_each=True)

        # Concatenate the original and conditional samples grids
        concatenated_grid = torch.cat((original_images_grid, conditional_images_grid), 2)  # Concatenate side by side

        # Log the concatenated grid to TensorBoard
        tag = f'gamma={gamma}'
        logger.experiment.add_image(tag, concatenated_grid)