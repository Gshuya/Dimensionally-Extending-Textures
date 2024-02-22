import torch
import torch.nn as nn



def sample_trilinear(noise, coords, img_h=128, img_w=128):
    """
    Samples noise octaves in one shot

    :param noise: noise cube [n_octaves, noise_res, noise_res, noise_res] (same noise foreach sample in batch)
    :param coords: octave-transformed sampling positions [bs, n_octaves, 3, img_h*img_w]
    :param img_h: height of image to synthesize
    :param img_w: width of image to synthesize
    :return: sampled noise octaves [bs, n_octaves, img_h, img_w]  
    """
  
    n_octaves, noise_res = noise.shape[:2]
    bs = coords.shape[0]
  
    # all contributing source coordinates (interpolation endpoints)
    x0 = torch.floor(coords[:, :, 0, :])
    x1 = x0 + 1
    y0 = torch.floor(coords[:, :, 1, :])
    y1 = y0 + 1
    z0 = torch.floor(coords[:, :, 2, :])
    z1 = z0 + 1

    # interpolation weights
    w_x = coords[:, :, 0, :] - x0
    w_y = coords[:, :, 1, :] - y0
    w_z = coords[:, :, 2, :] - z0
    
    # mod for out-of-bound indices
    x0 = torch.fmod(x0, torch.ones_like(x0) * noise_res)
    x1 = torch.fmod(x1, torch.ones_like(x1) * noise_res)
    y0 = torch.fmod(y0, torch.ones_like(y0) * noise_res)
    y1 = torch.fmod(y1, torch.ones_like(y1) * noise_res)
    z0 = torch.fmod(z0, torch.ones_like(z0) * noise_res)
    z1 = torch.fmod(z1, torch.ones_like(z1) * noise_res)

    noise = torch.reshape(noise, [n_octaves, noise_res**3])
 

    idx_x0_y0_z0 = ((x0 * noise_res**2 + y0 * noise_res + z0) % noise.shape[1]).long()
    idx_x0_y0_z1 = ((x0 * noise_res**2 + y0 * noise_res + z1) % noise.shape[1]).long()
    idx_x0_y1_z0 = ((x0 * noise_res**2 + y1 * noise_res + z0) % noise.shape[1]).long()
    idx_x0_y1_z1 = ((x0 * noise_res**2 + y1 * noise_res + z1) % noise.shape[1]).long()
    idx_x1_y0_z0 = ((x1 * noise_res**2 + y0 * noise_res + z0) % noise.shape[1]).long()
    idx_x1_y0_z1 = ((x1 * noise_res**2 + y0 * noise_res + z1) % noise.shape[1]).long()
    idx_x1_y1_z0 = ((x1 * noise_res**2 + y1 * noise_res + z0) % noise.shape[1]).long()
    idx_x1_y1_z1 = ((x1 * noise_res**2 + y1 * noise_res + z1) % noise.shape[1]).long()
  

    def batched_gather(idx):
        out = []
        for i in range(n_octaves):
            bg=[] 
            for j in range(bs):
                g = torch.gather(noise[i], dim=0, index=idx[j, i, :])
                bg.append(g)
            out.append(torch.unsqueeze(torch.stack(bg), 1))
        return torch.cat(out, 1)
    
   
    # gather contributing samples
    val_x0_y0_z0 = batched_gather(idx_x0_y0_z0)
    val_x0_y0_z1 = batched_gather(idx_x0_y0_z1)
    val_x0_y1_z0 = batched_gather(idx_x0_y1_z0)
    val_x0_y1_z1 = batched_gather(idx_x0_y1_z1)
    val_x1_y0_z0 = batched_gather(idx_x1_y0_z0)
    val_x1_y0_z1 = batched_gather(idx_x1_y0_z1)
    val_x1_y1_z0 = batched_gather(idx_x1_y1_z0)
    val_x1_y1_z1 = batched_gather(idx_x1_y1_z1)

    # Interpolate along z 
    c_00 = val_x0_y0_z0 * (1.0 - w_z) + val_x0_y0_z1 * w_z
    c_01 = val_x0_y1_z0 * (1.0 - w_z) + val_x0_y1_z1 * w_z
    c_10 = val_x1_y0_z0 * (1.0 - w_z) + val_x1_y0_z1 * w_z
    c_11 = val_x1_y1_z0 * (1.0 - w_z) + val_x1_y1_z1 * w_z

    # Interpolate along y 
    c_0 = c_00 * (1.0 - w_y) + c_01 * w_y
    c_1 = c_10 * (1.0 - w_y) + c_11 * w_y

    # Interpolate along x
    c = c_0 * (1.0 - w_x) + c_1 * w_x

    # reshape
    c = c.view(bs, n_octaves, img_h, img_w)
    return c


class add_noise(nn.Module):
    """
    Add noise in 4 layers. In each layer, add 4 (n_octaves=16) or 8 (n_octaves=32) octaves
    :param t_in: pre-activated layer output
    :param noise: noise octaves
    :param layer_idx: layer index
    :return: pre-activated layer output with added noise octaves
    """
    def __init__(self, layer_idx, hidden_dim, n_octaves):
        super(add_noise, self).__init__()
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        self.per_octave_w16 = nn.Parameter(nn.init.normal_(torch.randn(4, self.hidden_dim)))
        self.per_octave_w32 = nn.Parameter(nn.init.normal_(torch.randn(8, self.hidden_dim)))
        self.n_octaves = n_octaves

    def forward(self, t_in, noise):
        t_out = t_in
        if self.n_octaves == 16:
            for i in range(4):
                t_out = t_out + noise[:, self.layer_idx * 4 + i:self.layer_idx * 4 + i + 1, :, :] * torch.reshape(self.per_octave_w16[i, :], [1, self.hidden_dim, 1, 1])
        elif self.n_octaves == 32:
            for i in range(8):
                t_out = t_out + noise[:, self.layer_idx * 8 + i:self.layer_idx * 8 + i + 1, :, : ] * torch.reshape(self.per_octave_w32[i, :], [1, self.hidden_dim, 1, 1])
        else:
            import sys
            sys.exit("oops, need n_octaves either 16 or 32!")
        

        return t_out
    


class bias(nn.Module):
    """
    Bias layer for adding bias after adding noise in each layer
    """
    def __init__(self, hidden_dim):
        super(bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
    
    def forward(self, t_in):
        t_out = t_in + self.bias

        return t_out



class Generator(torch.nn.Module):

    """
    MLP that maps 3D coordinate to an rgb texture value

    noise_cube: noise, same foreach sample in batch [n_octaves, noise_res, noise_res, noise_res]
    coords: xyz coords of slices to synthesize [bs, 3, img_size^2)
    transformations: octave transformations [bs, n_octaves, 3, 3]
    :param img_size: training patch size
    :param img_h: height of output image (for inference)
    :param img_w: width of output image (for inference)
    :param hidden_dim: number of neurons in dense layers

    :return: rgb image
    """

    def __init__(self, img_size=128, img_h=None, img_w=None, hidden_dim=128, n_octaves=16):
        super(Generator, self).__init__()

        # Image size
        self.img_size = img_size
        self.img_h = img_h 
        self.img_w = img_w

        if self.img_h is None:
            self.img_h = self.img_size
        if self.img_w is None:
            self.img_w = self.img_size
        
        self.n_octaves = n_octaves

        # Define constant input layer 
        self.base = nn.Parameter(torch.ones(1, hidden_dim, 1, 1))

        # Define transformation to learn n noise frequencies
        self.transformations = nn.Parameter((torch.randn(n_octaves, 3, 3)).unsqueeze(0))
        
        
        # Define learning layers 
        self.conv0 = nn.Conv2d(hidden_dim, hidden_dim,kernel_size=1, bias=False)
        self.bias0 = bias(hidden_dim)
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim,kernel_size=1, bias=False)
        self.bias1 = bias(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim,kernel_size=1, bias=False)
        self.bias2 = bias(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim,kernel_size=1, bias=False)
        self.bias3 = bias(hidden_dim)
        
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim,kernel_size=1)
        self.conv5 = nn.Conv2d(hidden_dim, 3,kernel_size=1) # 3 rgb channels
        self.noise0 = add_noise(0, hidden_dim, n_octaves)
        self.noise1 = add_noise(1, hidden_dim, n_octaves)
        self.noise2 = add_noise(2, hidden_dim, n_octaves)
        self.noise3 = add_noise(3, hidden_dim, n_octaves)
        self.relu = nn.LeakyReLU(0.2, inplace=True)


    def trans(self, n_octaves, trans, batch_size):
            """Returns octave transformations"""

            # broadcast to entire batch [bs, n_octaves, 3, 3]
            trans = trans.repeat(batch_size, 1, 1, 1) 

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            oct_factors = torch.zeros((n_octaves, 3, 3), dtype=torch.float32).to(device)
            for i in range(n_octaves):
                for j in range(3):
                    if n_octaves == 16:
                        oct_factors[i, j, j] = 2 ** i
                    else:
                        oct_factors[i, j, j] = 2 ** (i/2.)
           
            # Apply octave transformations
            transformations = torch.matmul(trans, oct_factors)
            return transformations
    


    def forward(self, coords, noise_cube):

        # broadcast (constant per pixel and per sample)
        BatchSize = coords.shape[0]
        base = self.base.expand(BatchSize, -1, self.img_h, self.img_w)
       
        # obtain octave sampling coordinates by transforming slice coordinates with octave matrices
        coords = torch.unsqueeze(coords, 1)  
        # broadcast (same input coords foreach octave)
        coords = coords.repeat(1, self.n_octaves, 1, 1)

        octave_transformations = self.trans(self.n_octaves, self.transformations, BatchSize)
        # A distinct set of coordinates foreach batch and octave
        octave_coords = torch.matmul(octave_transformations, coords) 
        sampled_noise = sample_trilinear(noise_cube, octave_coords, img_h=self.img_h, img_w=self.img_w)
       
    
        l0 = self.conv0(base)
        l0 = self.noise0(l0, sampled_noise)
        l0 = self.bias0(l0)
        l0 = self.relu(l0)

        l1 = self.conv1(l0)
        l1 = self.noise1(l1, sampled_noise)
        l1 = self.bias1(l1)
        l1 = self.relu(l1)

        l2 = self.conv2(l1)
        l2 = self.noise2(l2, sampled_noise)
        l2 = self.bias2(l2)
        l2 = self.relu(l2)

        l3 = self.conv3(l2)
        l3 = self.noise3(l3, sampled_noise)
        l3 = self.bias3(l3)
        l3 = self.relu(l3)

        l4 = self.conv4(l3)
        l4 = self.relu(l4)

        rgb = self.conv5(l4)
        
        return rgb 

        






class Discriminator(nn.Module):

    """
    Discriminator that takes an image and 
    returns a probability of it being real
    """
    def __init__(self, input_shape) -> None:
      
        super(Discriminator, self).__init__()
        bs, channels, height, width = input_shape

        def discriminator_block(in_filters, out_filters):
            """Returns layers of each discriminator block"""
            layers = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, bias=True, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
            return layers

        def fc_block(in_filters, out_filters):
            """Returns fc layers"""
            layers = nn.Sequential(
                nn.Linear(in_filters, out_filters, bias=True),
                nn.LeakyReLU(0.2, inplace=True)
            )
            return layers

        # Average pool downsampling layer
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.l0 = discriminator_block(channels, 32) 
        self.l1 = discriminator_block(32, 64)
        self.l2 = discriminator_block(64, 128)
        self.l3 = discriminator_block(128, 256)
        self.l4 = discriminator_block(256, 256)
        self.l5 = discriminator_block(256, 512)
        self.l6 = discriminator_block(512, 512)
        self.fc = fc_block(512, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, img):
        l0 = self.l0(img)
       
        l1 = self.downsample(l0)
        l1 = self.l1(l1) 
       
        l2 = self.downsample(l1)
        l2 = self.l2(l2)
        
        l3 = self.downsample(l2)
        l3 = self.l3(l3)
        
        l4 = self.downsample(l3)
        l4 = self.l4(l4) 
        
        l5 = self.downsample(l4)
        l5 = self.l5(l5) 
        
        l6 = self.downsample(l5)
        l6 = self.l6(l6) 
        
        l7 = self.downsample(l6)
        l7 = torch.reshape(l7, [-1, 512])
        l7 = self.fc(l7) 
       
        logits = self.fc2(l7) 

        return logits, [l0, l1, l2, l3, l4, l5, l6]



    
