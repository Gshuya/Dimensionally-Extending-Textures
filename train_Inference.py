from Utils import meshgrid2D, get_random_slicing_matrices
import Models

import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch.optim as optim
import torch.autograd as autograd


batch_size = 16 
hidden_dim = 128 #how many neurons in hidden layers of sampler
beta = 1.  #weight for optional (beta>0) style loss
alpha = .1 #weight for optional (alpha>0) GAN loss
dis_iter = 1 #how many discriminator updates per generator update
progress_interval = 10000  #dump weights every n steps
d_lr = 2e-3 #learning rate D
g_lr = 5e-4 #learning rate G
gpW = 10 #gradient penalty weight
chpt_dir = "./tf/chpts" #where to store checkpoints
#sum_dir = "./tf/summaries" #where to store tensorboard summaries
img_size = 128 #training image patch size
noise_resolution = 64  #w, h, d of noise solid
n_octaves = 16   #how many noise octaves to use. Needs to be either 16 or 32
training_exemplar = './exemplars/0.png' #path to the training exemplar
wood = False #whether to use only specific slices, e.g., for wood, or completely random

num_samples = 1 #how many samples to generate
img_w = 512 #output image patch size
img_h = 512 #output image patch size
seed = -1 #seed for reproducibility, -1 for random
model_name = "./tf/chpts/checkpoint_s100000" #checkpoint file to load file to load
out_path = './outputs/' #path to output image(s)

torch.cuda.empty_cache()

def gradient_penalty(real, fake, f):
    def interpolate(a, b):
        shape = torch.cat(
            (torch.shape(a)[0:1], torch.tile([1], [a.shape.ndims - 1])),
            axis=0)
        alpha = torch.rand(shape=shape)
        inter = a + alpha * (b - a)
        inter.set_shape(a.get_shape().as_list())
        return inter

    x = interpolate(real, fake)
    pred, _ = f(x)
    gradients = torch.autograd.grad(pred, x)[0]
    slopes = torch.sqrt(torch.sum(
        torch.square(gradients),
        axis=list(range(1, len(x.shape)))))
    gp = torch.mean((slopes - 1.) ** 2)

    return gp



def style_loss(feat_real, feat_fake):
    def gram_matrix(t):
        einsum = torch.einsum('bijc,bijd->bcd', t, t)
        n_pix = t.get_shape().as_list()[1]*t.get_shape().as_list()[2]
        return einsum/n_pix

    real_gram_mat = []
    fake_gram_mat = []

    # all D features except logits
    for i in range(len(feat_real)):
        real_gram_mat.append(gram_matrix(feat_real[i]))
        fake_gram_mat.append(gram_matrix(feat_fake[i]))

    # l1 loss
    style_loss = torch.add_n([torch.reduce_mean(torch.abs(real_gram_mat[idx] - fake_gram_mat[idx]))
                           for idx in range(len(feat_real))])

    return style_loss / len(feat_real)


def get_real_imgs(img):
    img_batch = []
    for i in range(batch_size):
        transform = transforms.Compose([
            transforms.RandomCrop((img_size, img_size), pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        img_crop = transform(img)
        #transpose to channels last
        img_crop = np.transpose(img_crop, [1, 2, 0])
        img_batch.append(torch.unsqueeze(img_crop, 0)) #add batch dim
    img_batch = torch.cat(img_batch, 0)
    return img_batch


"""Training"""
def train():
    #model to device GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    # load training exemplar
    training_img = cv2.imread(training_exemplar).astype(np.float32) / 255.
    training_img = cv2.cvtColor(training_img, cv2.COLOR_BGR2RGB)
    training_img = np.transpose(training_img, [2, 0, 1])
    # convert to tensor
    training_img = torch.from_numpy(training_img)
    print('training_img.shape:', training_img.size())
    real_batch_op = get_real_imgs(training_img).to(device)
    print('real_batch_op.shape:', real_batch_op.size())
   

    """
    TRANSFORMER: in single exemplar, we directly optimize for the transformation parameters
    """
    transformations = torch.nn.Parameter(torch.randn(n_octaves, 3, 3)).clone().detach()
    transformations = transformations.to(device)
    # broadcast to entire batch
    transformations = transformations.unsqueeze(0)
    transformations = transformations.repeat(batch_size, 1, 1, 1)
    transformations.float()
    
    print(transformations.is_leaf)
    
    transformations = transformations.requires_grad_(True)
    print(transformations.is_leaf)
    print(transformations.device)

    # Define the models
    discriminator = Models.Mdiscriminator()
    sampler = Models.Sampler()
    discriminator.float()
    sampler.float()
    sampler.requires_grad_(True)

    

    #print('sampler.parameters():', list(sampler.parameters()))
    #print('dis.parameters():', list(discriminator.parameters()))
    #print('transformations.parameters():', [transformations])

    # Define the optimizer for each model]
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
    
    g_optimizer = optim.Adam(list(sampler.parameters())+[transformations], lr=g_lr, betas=(0.5, 0.999))
   
    
    # Initialize the summary writer
    #summary_writer = SummaryWriter(sum_dir)
    
    # Training loop
    step = 0
    while True:
        # Train discriminator
        for i in range(dis_iter):
            discriminator.zero_grad()
            
            # Get random slicing matrices
            slicing_matrix_ph = get_random_slicing_matrices(batch_size, wood=wood)
            slicing_matrix_ph = torch.from_numpy(slicing_matrix_ph).float()
            

            # Get some noise
            # Using the same noise instance foreach sample in batch for efficiency
            octaves_noise_ph = torch.randn([n_octaves, noise_resolution, noise_resolution, noise_resolution]).to(device)
            
            """
            SAMPLER
            """
            # single slice at z=0 [4, img_size^2]
            coords = meshgrid2D(img_size, img_size)
            coords.float()
            # bs different random slices [bs, img_size^2]
            coords = torch.matmul(slicing_matrix_ph, coords)
            # drop homogeneous coordinate
            coords = coords[:, :3, :]
            coords = coords.to(device)

            # Generate fake images
            print(coords.shape)
            print(transformations.shape)
            print(octaves_noise_ph.shape)

            fake_img = sampler(octaves_noise_ph, coords, transformations)
            
            # Get logits
            logits_fake, feat_fake = discriminator(fake_img, reuse=False, scope='discriminator', act=torch.nn.LeakyReLU, eq_lr=True)
            logits_real, feat_real = discriminator(real_batch_op, reuse=True, scope='discriminator', act=torch.nn.LeakyReLU, eq_lr=True)

            
            # Compute the D loss
            wgan_d_loss = torch.mean(logits_fake) - torch.mean(logits_real)
            gp = gradient_penalty(real_batch_op, fake_img, discriminator)
            d_loss = wgan_d_loss + gpW * gp + 0.001 * torch.pow(logits_real, 2)
            
            d_loss.backward()
            d_optimizer.step()
            #d_optimizer.minimize(d_loss)
            """
            if step % 100 == 0 and i == 0:
                # Write discriminator summaries
                summary_writer.add_scalar('logits_real', torch.mean(logits_real), step)
                summary_writer.add_scalar('logits_fake', torch.mean(logits_fake), step)
                summary_writer.add_scalar('d_loss', wgan_d_loss, step)
                summary_writer.add_scalar('gp', gp, step)
            """
        # Train generator
        g_optimizer.zero_grad()
        
        # Compute the generator loss
        g_style = (beta != 0.) #weight for optional (beta>0) style loss
        g_gan = (alpha != 0)  #weight for optional (alpha>0) GAN loss

        g_gan_loss = -torch.mean(logits_fake)
        g_style_loss = style_loss(feat_real, feat_fake)

        if alpha < 0 and beta < 0:
            raise Exception("oops, must do either alpha or beta > 0!")
        g_loss = alpha * g_gan_loss + beta * g_style_loss
        
        g_loss.backward()
        g_optimizer.step()
        """
        if step % 100 == 0:
            # Write generator summaries
            summary_writer.add_image('fake_img', torch.cat([real_batch_op, fake_img], dim=2).clamp(0., 1.), step)
            if g_gan:
                summary_writer.add_scalar('g_gan_loss', g_gan_loss, step)
            if g_style:
                summary_writer.add_scalar('g_style_loss', g_style_loss, step)
        """ 
        step += 1
        
        # Store checkpoint
        if step % progress_interval == 0:
            torch.save({
                'step': step,
                'discriminator_state_dict': discriminator.state_dict(),
                'sampler_state_dict': sampler.state_dict(),
                'transformations_state_dict': transformations,
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
            }, chpt_dir + '/checkpoint_s' + str(step))
        

"""Inference"""

def inference():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # setting seed for reproducibility
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print("setting seed for reproducibility to " + str(seed))

    # noise op
    s = noise_resolution
    octaves_noise_ph = torch.randn(n_octaves, s, s, s)
    octaves_noise_ph = octaves_noise_ph.to(device)

    """
    TRANSFORMER: in single exemplar, we directly optimize for the transformation parameters
    """
    transformations = torch.nn.Parameter(torch.randn(n_octaves, 3, 3))
    # broadcast to entire batch
    transformations = transformations.unsqueeze(0)
    transformations = transformations.to(device)

    """
    SAMPLER
    """
    # single slice at z=0 [4, img_h*img_w]
    coords = meshgrid2D(img_h, img_w)
    # random slicing
    slicing_matrix_ph = get_random_slicing_matrices(1)
    coords = torch.matmul(slicing_matrix_ph, coords)
    # drop homogeneous coordinate
    coords = coords[:, :3, :]
    coords = coords.to(device)

    sampler = Models.Sampler

    # load model
    checkpoint = torch.load(model_name)
    sampler.load_state_dict(checkpoint['sampler_state_dict'])
    transformations.data = checkpoint['transformations']

    print('loaded model from: ' + model_name)

    fake_img = sampler(octaves_noise_ph, coords, transformations, img_h=img_h, img_w=img_w,
                 act=torch.nn.LeakyReLU(), scope='sampler', hidden_dim=hidden_dim, eq_lr=True)


    for i in range(num_samples):
        # get a random slicing matrix
        slicing_matrix_ph = get_random_slicing_matrices(1) #????

        # get some noise
        octaves_noise_ph = torch.randn(n_octaves, s, s, s)
        octaves_noise_ph = octaves_noise_ph.to(device)

        # get batch_size samples passing slicing_matrix_ph and octaves_noise_ph
        
        fake_img = sampler(octaves_noise_ph, coords, transformations, img_h=img_h, img_w=img_w,act=torch.nn.LeakyReLU(), scope='sampler', hidden_dim=hidden_dim, eq_lr=True) 
        imgs = fake_img.detach().cpu().numpy()
        img = imgs[0]
        cv2.imwrite(out_path + str(i+1) + '.png', cv2.cvtColor(np.clip(img, 0., 1.) * 255, cv2.COLOR_BGR2RGB),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
        

def main():
    train()
    inference()

if __name__ == "__main__":
    main()


        



    
  


