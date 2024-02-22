from Utils import meshgrid2D, get_random_slicing_matrices
import Models
import Config

import torch
import numpy as np
from torchvision.utils import save_image



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
    generator.to(device)


    # Generate images
    for i in range(cfg.num_samples):
        # get random slices
        slicing_matrix_ph = get_random_slicing_matrices(4, random=cfg.random) 
        slicing_matrix_ph = torch.from_numpy(slicing_matrix_ph).float()
        coords = meshgrid2D(cfg.img_h, cfg.img_w)  
        coords = torch.matmul(slicing_matrix_ph, coords)
        coords = coords[:, :3, :]   
        coords = coords.to(device)
        # Generate random noise
        noise_cube = torch.randn([cfg.n_octaves, cfg.noise_resolution, cfg.noise_resolution, cfg.noise_resolution])
        noise_cube = noise_cube.to(device)
      
        # Generate images
        imgs = generator(coords,noise_cube)
    
        save_image(imgs, cfg.out_path + str(i+1) + '.png')
        print('Image saved to: ' + cfg.out_path + str(i+1) + '.png')
  


################################
if __name__ == "__main__":
    cfg = Config.Config()
    inference()
