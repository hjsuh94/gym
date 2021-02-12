import numpy as np
import torch

def lyapunov_measure():
    """
    Return lyapunov measure by creating a weighted matrix.
    """
    pixel_radius = 7
    measure = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            radius = np.linalg.norm(np.array([i - 15.5, j - 15.5]), ord=2) ** 2.0
            measure[i,j] = np.maximum(radius - pixel_radius, 0)
    return measure

def lyapunov(image, device="cpu"):
    """
    Apply the lyapunov measure to the image. Expects (B x 1 x 32 x 32), output B vector.
    """
    V_measure = torch.Tensor(lyapunov_measure()).to(device)
    return torch.sum(torch.mul(image, V_measure), [2,3])
