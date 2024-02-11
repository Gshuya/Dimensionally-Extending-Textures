class Config: 
    # training parameters
    img_size = 128  # training image patch size
    batch_size = 16 
    hidden_dim = 128 
    epoch = 800
    d_lr = 2e-3  # discriminator's learning rate
    g_lr = 5e-4  # generator's learning rate
    beta = 1.  #weight for optional (beta>0) style loss
    alpha = .1 #weight for optional (alpha>0) GAN loss
    lambda_gp = 10 # weight for gradient penalty
    noise_resolution = 64  # w, h, d of noise solid
    n_octaves = 16  # Noise octaves, needs to be either 16 or 32
    training_exemplar = './exemplars/bumpy_0059.jpg' # path to the training exemplar
    chpt_path = "./chpts/bumpy/" # where to store checkpoints
    random = True # whether to use only specific slices
    chpt = "./chpts/bumpy/final_model.pt"

    # Inference parameters
    num_samples = 1 # how many samples to generate
    img_w = 128 # output image patch size
    img_h = 128 # output image patch size
    seed = -1 # seed for reproducibility, -1 for random
    model_name ="./chpts/bumpy/checkpoint_epoch_200.pt" # load saved model for inference
    out_path = './outputs/bumpy/200' 

