import torch
import torch.nn as nn
import numpy as np

def power_iteration(A, num_simulations):
    b_k = np.random.rand(A.shape[1])
    for _ in range(num_simulations):
        # Calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)
        
        # Re normalize the vector
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm

    return b_k, b_k1_norm

def inverse_power_iteration(A, num_simulations):
    b_k = np.random.rand(A.shape[1])
    for _ in range(num_simulations):
        # Solve the system of linear equations for inverse iteration
        b_k1 = np.linalg.solve(A, b_k)
        
        # Re normalize the vector
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm

    return b_k, b_k1_norm

def gram_matrix_condition_number_iterative(A, num_simulations=1000):
    # Gram matrix
    G = np.dot(A.T, A)
    
    # Estimate the largest singular value
    _, sigma_max = power_iteration(G, num_simulations)
    
    # Estimate the smallest singular value using inverse power iteration
    _, sigma_min = inverse_power_iteration(G, num_simulations)
    
    # Compute the condition number of the Gram matrix
    condition_number = sigma_max / sigma_min
    
    return condition_number

def gram_matrix_condition_number_svd(A):    
    # Compute the Singular Value Decomposition of A using PyTorch
    U, singular_values, V = torch.linalg.svd(A)
    
    # Compute the condition number of the Gram matrix
    condition_number = (singular_values.max() / singular_values.min()).item()**2
    
    return condition_number

def convert_to_one_hot(labels):
    # Get the device of the input labels tensor
    device = labels.device
    
    # Convert each 0 and 1 to a 2-dimensional one-hot encoding
    one_hot = torch.zeros((labels.size(0), labels.size(1) * 2), device=device)
    
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

def to_numpy(x):
    """Convert Pytorch tensor to numpy array."""
    return x.clone().detach().cpu().numpy()


class HSIC(nn.Module):
    """
    Base class for the finite sample estimator of Hilbert-Schmidt Independence Criterion (HSIC)
    ..math:: HSIC (X, Y) := || C_{x, y} ||^2_{HS}, where HSIC (X, Y) = 0 iif X and Y are independent.

    Parameters
    ----------
    kernel_x : callable
        The kernel function for domain X.
    kernel_y : callable
        The kernel function for domain Y.
    algorithm: str ('unbiased' / 'biased')
        The algorithm for the finite sample estimator. 'unbiased' is used for our paper.
    """
    def __init__(self, kernel_x, kernel_y, algorithm='unbiased'):
        super(HSIC, self).__init__()

        self.kernel_x = kernel_x
        self.kernel_y = kernel_y

        if algorithm == 'biased':
            self.estimator = self.biased_estimator
        elif algorithm == 'unbiased':
            self.estimator = self.unbiased_estimator
        else:
            raise ValueError('invalid estimator: {}'.format(algorithm))

    def biased_estimator(self, input1, input2):
        """Biased estimator of Hilbert-Schmidt Independence Criterion."""
        K = self.kernel_x(input1)
        #print(K[:5, :5])
        L = self.kernel_y(input2)

        KH = K - K.mean(0, keepdim=True)
        LH = L - L.mean(0, keepdim=True)

        N = len(input1)

        return torch.trace(KH @ LH / (N - 1) ** 2)

    def unbiased_estimator(self, input1, input2):
        """Unbiased estimator of Hilbert-Schmidt Independence Criterion."""
        kernel_XX = self.kernel_x(input1)
        kernel_YY = self.kernel_y(input2)

        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)

        N = len(input1)

        hsic = (
            torch.trace(tK @ tL)
            + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
            - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )

        return hsic / (N * (N - 3))

    def forward(self, input1, input2, **kwargs):
        return self.estimator(input1, input2)


def rbf_kernel(X, sigma):
    """
    Radial Basis Function (RBF) kernel.
    """
    X = X.view(len(X), -1)
    XX = X @ X.t()
    X_sqnorms = torch.diag(XX)
    X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
    gamma = 1 / (2 * sigma ** 2)
    #print(X_L2[:5,:5])
    kernel_XX = torch.exp(-gamma * X_L2)
    return kernel_XX


def dot_product_kernel(X):
    """
    Simple dot product kernel.
    """
    X = X.view(len(X), -1)
    return X @ X.t()


# Example usage:
# Create an instance of HSIC with RBF kernel for domain X and dot product kernel for domain Y
#sigma_x = 1.0
#hsic_instance = HSIC(kernel_x=lambda X: rbf_kernel(X, sigma_x), kernel_y=kernel_y=lambda Y: dot_product_kernel(Y), algorithm='unbiased')
