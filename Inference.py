from Utils import meshgrid2D, get_random_slicing_matrices
import Models
import Config

import torch
import numpy as np
import cv2
from torchvision.utils import save_image
#import matplotlib.pyplot as plt

#import configuration
cfg = Config.Config()

"""Inference"""

def inference():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # setting seed for reproducibility
    if cfg.seed != -1:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        print("setting seed for reproducibility to " + str(cfg.seed))

    # Instantiate the generator
    generator = Models.Generator(img_size=cfg.img_size, img_h=cfg.img_h, img_w=cfg.img_w, hidden_dim=cfg.hidden_dim, n_octaves=cfg.n_octaves)
   
    # Load the saved model
    checkpoint = torch.load(cfg.model_name)
    # epoch = checkpoint['epoch']
    generator.load_state_dict(checkpoint['generator_state_dict'])
    # print('loaded model from: ' + cfg.model_name)
    generator.to(device)

    # Generate samples
    for i in range(cfg.num_samples):
        # get random noise
        slicing_matrix_ph = get_random_slicing_matrices(1, random=cfg.random) # single slice [1, 4, 4]
        slicing_matrix_ph = torch.from_numpy(slicing_matrix_ph).float()
        coords = meshgrid2D(cfg.img_h, cfg.img_w)  # [4, img_h*img_w]
        coords = torch.matmul(slicing_matrix_ph, coords) # [1, 4, img_h*img_w]
        coords = coords[:, :3, :]   
        coords = coords.to(device)
        # print("coords.shape:", coords.shape)
        noise_cube = torch.randn([cfg.n_octaves, cfg.noise_resolution, cfg.noise_resolution, cfg.noise_resolution])
        noise_cube = noise_cube.to(device)
      
        imgs = generator(coords,noise_cube)
        # print(imgs.shape)

        save_image(imgs, cfg.out_path + str(i+1) + '.png')
        print('Image saved')
  
################################
inference()
