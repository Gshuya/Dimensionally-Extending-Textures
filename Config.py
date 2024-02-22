class Config: 
    # training parameters

    
    img_size = 128  # patch size
    batch_size = 16 
    hidden_dim = 128 
    epoch = 100000  
    d_lr = 2e-3  # discriminator's learning rate
    g_lr = 5e-4  # generator's learning rate
    beta = 1.  # weight for optional (beta>0) style loss
    alpha = 0.1 # weight for optional (alpha>0) GAN loss
    lambda_gp = 10 # weight for gradient penalty
    noise_resolution = 64  # w, h, d of noise solid
    n_octaves = 16  # Noise octaves, needs to be either 16 or 32
    training_exemplar = './exemplars/wood.jpg' 
    chpt_path = "./chpts/wood/" # checkpoints
    random = False # whether to use only specific slices
    chpt = "./chpts/wood/checkpoint_epoch_158000.pt" # load saved model for training
    DperG = 1 # N. discriminator updates per generator update

    # Inference parameters
    num_samples = 1 # how many samples to generate
    img_w = 128 # output image 
    img_h = 128 # output image 
    seed = -1 # seed for reproducibility, -1 for random
    model_name ="./chpts/wood/checkpoint_epoch_60000.pt" # load saved model for inference
    out_path = './outputs/wood/60000' 


