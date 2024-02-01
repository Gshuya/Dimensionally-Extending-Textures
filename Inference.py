from Utils import meshgrid2D, get_random_slicing_matrices
import Models
import Config

import torch
import numpy as np
import cv2
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
    sampler = Models.Sampler(img_size=cfg.img_size, img_h=cfg.img_h, img_w=cfg.img_w, hidden_dim=cfg.hidden_dim, noise_resolution=cfg.noise_resolution, n_octaves=cfg.n_octaves, batch_size = cfg.batch_size)
   
    # Load the saved model
    checkpoint = torch.load(cfg.model_name)
    epoch = checkpoint['epoch']
    sampler.load_state_dict(checkpoint['generator_state_dict'])
    print('loaded model from: ' + cfg.model_name + '_' +epoch + 'epochs')
    sampler.to(device)

    
    for i in range(cfg.num_samples):
        # get random noise
        slicing_matrix_ph = get_random_slicing_matrices(1, random=cfg.random) # single slice [1, 4, 4]
        slicing_matrix_ph = torch.from_numpy(slicing_matrix_ph) #.float()
        coords = meshgrid2D(cfg.img_h, cfg.img_w)  # [4, img_h*img_w]
        coords = torch.matmul(slicing_matrix_ph, coords) # [1, 4, img_h*img_w]
        coords = coords[:, :3, :]   
        coords = coords.to(device)
      
        imgs = sampler(coords) 
        print(imgs.shape)

        # convert to numpy for visualization
        #generated_samples = (generated_samples.cpu().numpy() + 1) / 2.0
        #print(generated_samples.shape)

        cv2.imwrite(cfg.out_path + str(i+1) + '.png', cv2.cvtColor(np.clip(imgs, 0., 1.) * 255, cv2.COLOR_BGR2RGB),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
      
    # Visualize generated samples
    """
    fig, axes = plt.subplots(1, cfg.num_samples, figsize=(12, 2))
    for i, ax in enumerate(axes):
        ax.imshow(np.transpose(generated_samples[i], (1, 2, 0)))
        ax.axis('off')
    plt.show()
    """

################################
inference()
