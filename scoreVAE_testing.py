import torch
import torchvision
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import pickle
from hsic import HSIC, rbf_kernel, dot_product_kernel

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

def check_changing_attributes(pl_module, datamodule, logger):
    pl_module.configure_sde(pl_module.config)
    dataloader_iterator = iter(datamodule.val_dataloader())
    x, y = next(dataloader_iterator)
    x, y = x.to(pl_module.device), y.to(pl_module.device)
    
    attribute_to_index_map = datamodule.get_attribute_to_index_map()
    #attributes_to_flip = [None, 'Eyeglasses', 'Male', 'Smiling']
    attributes_to_flip = ['Wavy_Hair', 'Gray_Hair', 'Male', 'Smiling']
    
    num_rows = int(sqrt(x.size(0)))
    original_images_grid = torchvision.utils.make_grid(x, nrow=num_rows, normalize=True, scale_each=True)
    
    z, x_T = pl_module.encode(x, y, encode_x_T=False)

    for gamma in [1.05]:
        for attribute_to_flip in attributes_to_flip:
            flipped_y = pl_module.flip_attributes(y, attribute_to_flip, attribute_to_index_map)
            x_flipped = pl_module.decode(flipped_y, z, x_T, gamma)
            flipped_images_grid = torchvision.utils.make_grid(x_flipped, nrow=num_rows, normalize=True, scale_each=True)
            
            # Concatenate the original and conditional samples grids
            concatenated_grid = torch.cat((original_images_grid, flipped_images_grid), 2)  # Concatenate side by side
            tag = f'gamma:{gamma} - Original and Flipped Attributes ({attribute_to_flip})'
            logger.experiment.add_image(tag, concatenated_grid)

def generate_latents(pl_module, datamodule, logger):
    # Ensure the directory exists
    output_dir = logger.log_dir
    os.makedirs(output_dir, exist_ok=True)

    # Assuming pl_module and datamodule are already correctly setup and moved to the correct device
    dataloader = datamodule.val_dataloader()

    y_stack = []
    z_stack = []

    # Encode all images and collect attributes and encoded vectors
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y = batch
            x, y = x.to(pl_module.device), y.to(pl_module.device)
            z, _ = pl_module.encode(x, y, encode_x_T=False)
            y_stack.append(y.cpu())
            z_stack.append(z.cpu())

    # Concatenate all vectors into big stacks
    y_stack = torch.cat(y_stack, dim=0).numpy()
    z_stack = torch.cat(z_stack, dim=0).numpy()

    # Save the data as a numpy array
    data_path = os.path.join(output_dir, 'data.npy')
    np.save(data_path, {'labels': y_stack, 'encodings': z_stack})
    print(f'Saved labels and encodings to {data_path}')

def test_correlation_matrix(pl_module, datamodule, logger):
    # Ensure the directory exists
    output_dir = logger.log_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    data_path = os.path.join(output_dir, 'data.npy')
    data = np.load(data_path, allow_pickle=True).item()
    y_stack = data['labels']
    z_stack = data['encodings']

    # Calculate cross-correlation matrix between y_stack and z_stack
    correlation_matrix = np.corrcoef(y_stack.T, z_stack.T)[:y_stack.shape[1], y_stack.shape[1]:]

    print(f'Dimensions of the correlation matrix: {correlation_matrix.shape}')
    print(f'Mean of the correlation matrix: {np.mean(correlation_matrix)}')
    print(f'Standard Deviation of the correlation matrix: {np.std(correlation_matrix)}')

    # Save the correlation matrix as a numpy array
    matrix_file_path = os.path.join(output_dir, 'correlation_matrix.npy')
    np.save(matrix_file_path, correlation_matrix)
    print(f'Correlation matrix saved to {matrix_file_path}')

    # Plot the correlation matrix with consistent color scaling
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Cross-Correlation Matrix between Attributes (y) and Encoded Vectors (z)')
    plt.xlabel('Encoded Dimensions (z)')
    plt.ylabel('Attributes (y)')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

    # Plot histogram of the correlation matrix values
    plt.figure(figsize=(10, 8))
    plt.hist(correlation_matrix.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Correlation Matrix Values')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'correlation_histogram.png'))
    plt.close()

    # Optionally log the image to TensorBoard
    img = plt.imread(os.path.join(output_dir, 'correlation_matrix.png'))
    logger.experiment.add_image('Cross-Correlation Matrix', torchvision.transforms.ToTensor()(img))
    img_hist = plt.imread(os.path.join(output_dir, 'correlation_histogram.png'))
    logger.experiment.add_image('Correlation Histogram', torchvision.transforms.ToTensor()(img_hist))

def convert_to_one_hot(labels):
    # Convert each 0 and 1 to a 2-dimensional one-hot encoding
    one_hot = torch.zeros((labels.size(0), labels.size(1) * 2))
    for i in range(labels.size(1)):
        one_hot[:, 2*i] = (labels[:, i] == 0).float()
        one_hot[:, 2*i + 1] = (labels[:, i] == 1).float()
    return one_hot

def median_heuristic(X):
    """
    Calculate the median heuristic for choosing sigma in the RBF kernel.
    """
    with torch.no_grad():
        pairwise_dists = torch.cdist(X, X, p=2)
        median_dist = torch.median(pairwise_dists)
        sigma = median_dist.item()
    return sigma


def old_calculate_hsic(logger):
    output_dir = logger.log_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    data_path = os.path.join(output_dir, 'data.npy')
    data = np.load(data_path, allow_pickle=True).item()
    y_stack = torch.tensor(data['labels'])
    z_stack = torch.tensor(data['encodings'])

    print(z_stack[0,:10])

    # Transform labels to one-hot encoding and concatenate
    y_stack_one_hot = convert_to_one_hot(y_stack)

    # Calculate sigma using the median heuristic
    #sigma = median_heuristic(z_stack)
    #print(f'Selected sigma: {sigma}')
    sigma=25

    # Compute HSIC 
    hsic_instance = HSIC(kernel_x=lambda X: rbf_kernel(X, sigma=sigma), kernel_y=dot_product_kernel, algorithm='unbiased')
    hsic_value = hsic_instance(z_stack, y_stack_one_hot).item()
    print(f'HSIC value: {hsic_value}')

    hsic_file_path = os.path.join(output_dir, 'hsic_value.txt')
    with open(hsic_file_path, 'w') as f:
        f.write(f'HSIC value: {hsic_value}\n')
    print(f'HSIC value saved to {hsic_file_path}')


def calculate_hsic(logger):
    output_dir = logger.log_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    data_path = os.path.join(output_dir, 'data.npy')
    data = np.load(data_path, allow_pickle=True).item()
    y_stack = torch.tensor(data['labels'])
    z_stack = torch.tensor(data['encodings'])

    # Transform labels to one-hot encoding
    y_stack_one_hot = convert_to_one_hot(y_stack)

    # Calculate total number of data points
    N = len(y_stack)

    # Define sigma for the RBF kernel
    sigma = 25

    # Create HSIC instances for unbiased and biased estimates
    hsic_unbiased = HSIC(kernel_x=lambda X: rbf_kernel(X, sigma=sigma), kernel_y=dot_product_kernel, algorithm='unbiased')
    hsic_biased = HSIC(kernel_x=lambda X: rbf_kernel(X, sigma=sigma), kernel_y=dot_product_kernel, algorithm='biased')

    # Generate log-spaced sample sizes starting from 64
    sample_sizes = np.logspace(np.log10(64), np.log10(4096), num=10, dtype=int)
    
    unbiased_means = []
    unbiased_stds = []
    biased_means = []
    biased_stds = []

    for size in tqdm(sample_sizes):
        unbiased_hsic_values = []
        biased_hsic_values = []

        for _ in range(10):  # Perform multiple evaluations for averaging
            indices = np.random.choice(N, size, replace=False)
            z_sample = z_stack[indices]
            y_sample = y_stack_one_hot[indices]

            unbiased_hsic_values.append(hsic_unbiased(z_sample, y_sample).item())
            biased_hsic_values.append(hsic_biased(z_sample, y_sample).item())

        unbiased_means.append(np.mean(unbiased_hsic_values))
        unbiased_stds.append(np.std(unbiased_hsic_values))
        biased_means.append(np.mean(biased_hsic_values))
        biased_stds.append(np.std(biased_hsic_values))

    # Calculate the ratio of sample mean to sample standard deviation
    unbiased_ratios = np.array(unbiased_means) / np.array(unbiased_stds)
    biased_ratios = np.array(biased_means) / np.array(biased_stds)

    print(sample_sizes)
    print(unbiased_means)
    print(unbiased_stds)

    # Plotting
    plt.figure(figsize=(18, 6))

    # Sample mean plot
    plt.subplot(1, 3, 1)
    plt.plot(sample_sizes, unbiased_means, label='Unbiased HSIC Mean')
    plt.plot(sample_sizes, biased_means, label='Biased HSIC Mean')
    plt.xscale('log')
    plt.xlabel('Sample Size')
    plt.ylabel('HSIC Mean')
    plt.legend()
    plt.title('HSIC Sample Mean')

    # Sample standard deviation plot
    plt.subplot(1, 3, 2)
    plt.plot(sample_sizes, unbiased_stds, label='Unbiased HSIC Std')
    plt.plot(sample_sizes, biased_stds, label='Biased HSIC Std')
    plt.xscale('log')
    plt.xlabel('Sample Size')
    plt.ylabel('HSIC Standard Deviation')
    plt.legend()
    plt.title('HSIC Sample Standard Deviation')

    # Ratio of mean to standard deviation plot
    plt.subplot(1, 3, 3)
    plt.plot(sample_sizes, unbiased_ratios, label='Unbiased Mean/Std Ratio')
    plt.plot(sample_sizes, biased_ratios, label='Biased Mean/Std Ratio')
    plt.xscale('log')
    plt.xlabel('Sample Size')
    plt.ylabel('HSIC Mean/Std Ratio')
    plt.legend()
    plt.title('HSIC Mean/Std Ratio')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'hsic_evaluation_plots.png')
    plt.savefig(plot_path)
    plt.show()

    print(f'HSIC evaluation plots saved to {plot_path}')
