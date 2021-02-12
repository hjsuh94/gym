import numpy as np

import torch
from torchvision import transforms
from PIL import Image
import cv2

def image_to_tensor(image_cv2):
    """
    Convert cv2 image of HxW into a torch representation of 1 x 1 x H x W. 
    """
    image_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    # 1. cv2 to PIL 
    image = Image.fromarray(image_cv2)
    # 2. PIL to tensor.
    image = image_transform(image)
    # 3. tensor to batch tensor 
    image = torch.unsqueeze(image, 0)
    return image 

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

def lyapunov(image_cv2):
    """
    Apply the lyapunov measure to the image. Expects (B x 1 x 32 x 32), output B vector.
    """
    image_tensor = image_to_tensor(image_cv2)
    V_measure = torch.Tensor(lyapunov_measure())
    torch_lyapunov = torch.sum(torch.mul(image_tensor, V_measure), [2,3])
    return torch_lyapunov.numpy().squeeze()
