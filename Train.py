from Utils import meshgrid2D, get_random_slicing_matrices
import Models
import Config

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
import torch.autograd as autograd
#from torch.autograd import profiler
import wandb

#import configuration
cfg = Config.Config()


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="DL_Texture3D",
    
    # track hyperparameters and run metadata
    config={
    "architecture": "GAN",
    "dataset": "wood",
    "DiscriminatorUpdates": "5",
    "epochs": 35000-40000,
    }
)


def gradient_penalty(real_images, fake_images, discriminator):
    batch_size = real_images.size(0)
    device = real_images.device

    # Generate random epsilon
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)

    # Interpolate between real and fake images
    interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated_images.requires_grad_(True)

    # Calculate the critic scores for interpolated images
    scores, _ = discriminator(interpolated_images)

    # Compute the gradients of the scores with respect to the interpolated images
    gradients = autograd.grad(outputs=scores, inputs=interpolated_images,
                              grad_outputs=torch.ones_like(scores),
                              create_graph=True, retain_graph=True)[0]

    # Calculate the gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def gram_matrix(t):
    einsum = torch.einsum('bijc,bijd->bcd', t, t)
    n_pix = t.size(1) * t.size(2)
    return einsum/n_pix

def style_loss(hiddenL_real, hiddenL_fake):
    real_gram = []
    fake_gram = []

    # Calculate gram matrices for feature extracted from hidden layers
    for i in range(len(hiddenL_real)):
        real_gram.append(gram_matrix(hiddenL_real[i]))
        fake_gram.append(gram_matrix(hiddenL_fake[i]))

    # Style loss
    style_loss = torch.stack([torch.mean(torch.abs(real_gram[i] - fake_gram[i])) for i in range(len(hiddenL_real))]).sum()

    return style_loss / len(hiddenL_fake)


def get_batch_imgs(img):
    """Get batch of images from single image"""
    img_batch = []
    for i in range(cfg.batch_size):
        transform = transforms.Compose([
            transforms.RandomCrop((cfg.img_size, cfg.img_size), pad_if_needed=True),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]) #normalized to [-1,1]
        ])
        img_crop = transform(img)
        img_batch.append(torch.unsqueeze(img_crop, 0))
    img_batch = torch.cat(img_batch, 0) 
    return img_batch



"""Training"""
def train():
    #model to device GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  

    # # load training exemplar
    # training_img = cv2.imread(cfg.training_exemplar).astype(np.float32) / 255.  #normalized to [0,1]
    # training_img = cv2.cvtColor(training_img, cv2.COLOR_BGR2RGB)
    # #print('training_img.shape:', training_img.shape)
    # training_img = np.transpose(training_img, [2, 0, 1])
    # # convert to tensor
    # training_img = torch.from_numpy(training_img)
    # #print('training_img.shape:', training_img.shape)

    training_img = Image.open(cfg.training_exemplar, mode='r').convert('RGB')
    real_batch = get_batch_imgs(training_img).to(device)
    # print('real_batch.shape:', real_batch.shape) # [16, 3, 128, 128]
   

    # Initialize the models
    sampler = Models.Sampler(img_size=cfg.img_size, hidden_dim=cfg.hidden_dim, n_octaves=cfg.n_octaves)
    discriminator = Models.Discriminator(real_batch.shape)


    # Define the optimizer for each model
    g_optimizer = optim.Adam(sampler.parameters(), lr=cfg.g_lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=cfg.d_lr, betas=(0.5, 0.999))
    
    # Move models to the selected device
    sampler.to(device)
    discriminator.to(device)

    # Load checkpoint
    checkpoint = torch.load(cfg.chpt)

    # Load generator and discriminator states
    sampler.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    # Load optimizer states
    g_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])

    epoch_chpt = checkpoint['epoch']

    
    # Training loop
    for epoch in range(cfg.epoch):
         
        """
        Train discriminator
        """

        for _ in range(cfg.DperG):

            d_optimizer.zero_grad()

            # Get random slicing matrices
            slicing_matrix_ph = get_random_slicing_matrices(cfg.batch_size, random=cfg.random) #[16,4,4]
            slicing_matrix_ph = torch.from_numpy(slicing_matrix_ph).float()

            # single slice [4, img_size^2]
            coords = meshgrid2D(cfg.img_size, cfg.img_size) 
            # bs different random slices [bs, img_size^2]
            coords = torch.matmul(slicing_matrix_ph, coords) # [16, 4, 4] * [4, 4, 16384] = [16, 4, 16384]
            # drop homogeneous coordinate
            coords = coords[:, :3, :]   
            coords = coords.to(device)
            # print("coords.shape:", coords.shape)

            noise_cube = torch.randn([cfg.n_octaves, cfg.noise_resolution, cfg.noise_resolution, cfg.noise_resolution])
            noise_cube = noise_cube.to(device)


            # Generate fake images
            fake_img = sampler(coords, noise_cube)
        
                
            # Get logits
            logits_real, hiddenL_real = discriminator(real_batch)
            logits_fake, hiddenL_fake = discriminator(fake_img)

            # Compute the Discriminator loss 
            d_org_loss = torch.mean(logits_fake) - torch.mean(logits_real)
            gp = gradient_penalty(real_batch, fake_img, discriminator)
            # real = 0.001 * torch.mean(logits_real)**2
            
            d_loss = d_org_loss + cfg.lambda_gp * gp


            d_loss.backward()
            d_optimizer.step()

        
        """
        Train generator
        """
        g_optimizer.zero_grad()

        
        # Generate fake images
        fake_img = sampler(coords, noise_cube)

        logits_real, hiddenL_real = discriminator(real_batch)
        logits_fake, hiddenL_fake = discriminator(fake_img)

        # Compute the generator loss
        if cfg.alpha < 0 and cfg.beta < 0:
            raise Exception("oops, must do either alpha or beta > 0!")
        
        g_gan_loss = -torch.mean(logits_fake)
        g_style_loss = style_loss(hiddenL_real, hiddenL_fake)
        
        g_loss = cfg.alpha * g_gan_loss + cfg.beta * g_style_loss

        
        g_loss.backward()
        g_optimizer.step()

        tot_epoch = epoch + epoch_chpt
        # Print the losses
        print(f"Epoch_{tot_epoch+1}, Generator Loss: {g_loss:.4f}, Discriminator Loss: {d_loss:.4f}")

        # Log the losses
        wandb.log({"Generator Loss": g_loss, "Discriminator Loss": d_loss})
        # Log the weights
        wandb.log({"Generator Weights": sampler.state_dict(), "Discriminator Weights": discriminator.state_dict()})

        # Save the model at desired intervals
        if (epoch + 1) % 5000 == 0:
            torch.save({
                'epoch': tot_epoch+1,
                'generator_state_dict': sampler.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_optimizer_state_dict': g_optimizer.state_dict(),
                'discriminator_optimizer_state_dict': d_optimizer.state_dict(),
            }, f'{cfg.chpt_path}checkpoint_epoch_{tot_epoch + 1}.pt')

    Final_epoch = cfg.epoch + epoch_chpt

    # Save the final trained model
    torch.save({
        'epoch': Final_epoch,
        'generator_state_dict': sampler.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'generator_optimizer_state_dict': g_optimizer.state_dict(),
        'discriminator_optimizer_state_dict': d_optimizer.state_dict(),
    }, f'{cfg.chpt_path}final_model.pt')
    print('model saved')



################################
train()


        



    
  


