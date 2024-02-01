class Config: 
    # training parameters
    img_size = 128  # training image patch size
    batch_size = 16 
    hidden_dim = 128 
    epoch = 1 
    d_lr = 2e-3  # discriminator's learning rate
    g_lr = 5e-4  # generator's learning rate
    beta = 1.  #weight for optional (beta>0) style loss
    alpha = .1 #weight for optional (alpha>0) GAN loss
    lambda_gp = 10 # weight for gradient penalty
    noise_resolution = 64  # w, h, d of noise solid
    n_octaves = 16  # Noise octaves, needs to be either 16 or 32
    training_exemplar = './exemplars/wood.jpg' # path to the training exemplar
    chpt_path = "./chpts/" # where to store checkpoints
    random = False # whether to use only specific slices

    # Inference parameters
    num_samples = 1 # how many samples to generate
    img_w = 512 # output image patch size
    img_h = 512 # output image patch size
    seed = -1 # seed for reproducibility, -1 for random
    model_name = "./chpts/final_model.pt" # checkpoint file to load
    out_path = './outputs/' # path to output image

