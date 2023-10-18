

def train2():
    #model to device GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load training exemplar
    training_img = cv2.imread(training_exemplar).astype(np.float32) / 255.
    training_img = cv2.cvtColor(training_img, cv2.COLOR_BGR2RGB)
    training_img = np.transpose(training_img, [2, 0, 1])
    # convert to tensor
    training_img = torch.from_numpy(training_img)
    print('training_img.shape:', training_img.size())

    real_batch_op = get_real_imgs(training_img)
    print('real_batch_op.shape:', real_batch_op.size())

    
    # noise op. Using the same noise instance foreach sample in batch for efficiency
    s = noise_resolution
    noise_op = torch.randn([n_octaves, s, s, s])

    
    #slicing_matrix_ph = torch.zeros([batch_size, 4, 4])
    #octaves_noise_ph = torch.zeros([n_octaves, s, s, s])
    
    """
    TRANSFORMER: in single exemplar, we directly optimize for the transformation parameters
    """
    with torch.no_grad():
        transformations = torch.randn([n_octaves, 3, 3], requires_grad=True)

    # broadcast to entire batch
    transformations = transformations.unsqueeze(0)
    transformations = transformations.repeat(batch_size, 1, 1, 1)



    """
    SAMPLER
    """
    # single slice at z=0 [4, img_size^2]
    coords = meshgrid2D(img_size, img_size)
    # bs different random slices [bs, img_size^2]
    coords = torch.matmul(slicing_matrix_ph, coords)
    # drop homogeneous coordinate
    coords = coords[:, :3, :]

    S = Models.sampler_single
    fake_img = S(octaves_noise_ph, coords, transformations, img_size=img_size,
                    act=torch.nn.LeakyReLU, scope='sampler', hidden_dim=hidden_dim, eq_lr=True)
    

    
    """
    DISCRIMINATOR
    """
    D = Models.discriminator
    logits_fake, feat_fake = D(fake_img, reuse=False, scope='discriminator', act=torch.nn.LeakyReLU, eq_lr=True)
    logits_real, feat_real = D(real_batch_op, reuse=True, scope='discriminator', act=torch.nn.LeakyReLU, eq_lr=True)

    # D loss
    wgan_d_loss = torch.mean(logits_fake) - torch.mean(logits_real)
    gp = gradient_penalty(real_batch_op, fake_img, D)
    d_loss = wgan_d_loss + gp + 0.001 * logits_real**2

    # G loss
    g_style = (beta != 0.) #weight for optional (beta>0) style loss
    g_gan = (alpha != 0)  #weight for optional (alpha>0) GAN loss

    if g_style and g_gan:
        g_gan_loss = -torch.mean(logits_fake)
        g_style_loss = style_loss(feat_real, feat_fake)
    elif g_gan:
        g_gan_loss = -torch.mean(logits_fake)
        g_style_loss = 0.
    elif g_style:
        g_gan_loss = 0.
        g_style_loss = style_loss(feat_real, feat_fake)
    else:
        raise Exception("oops, must do either alpha or beta > 0!")
    
    g_loss = alpha * g_gan_loss + beta * g_style_loss

    # train steps
    d_vars = torch.nn.ModuleList(D.parameters())
    #d_train_step = torch.optim.Adam(d_vars, lr=d_lr, betas=(0.5, 0.999)).minimize(d_loss)

    #g_vars = torch.nn.ModuleList(S.parameters())
    #g_train_step = torch.optim.Adam(g_vars, lr=g_lr, betas=(0.5, 0.999)).minimize(g_loss)

    # training loop
    step = 0
    while True:
        # train D
        for i in range(dis_iter):
            # get random slicing matrices
            slicing_matrix_ph = get_random_slicing_matrices(batch_size, wood=wood)
            # get some noise
            octaves_noise_ph = noise_op

            if step % 100 == 0 and i == 0:
                run_results = d_train_step
                print('d_loss:', run_results)

            else:
                d_train_step

        # train G
        if step % 100 == 0:
            run_results = g_train_step
            print('g_loss:', run_results)
        else:
            g_train_step
        step += 1

        # store chpt
        if step % progress_interval == 0:
            torch.save(S.state_dict(), chpt_dir + '/checkpoint_s' + str(step))
            torch.save(D.state_dict(), chpt_dir + '/checkpoint_d' + str(step))
        
    



    
"""
    
def discriminator(img, reuse=True, act=torch.nn.LeakyReLU, scope='discriminator', eq_lr=True):
    
    def post_conv(t_in, layer_idx=-1):
      
        #Applies bias and activation function
     
        t_out = _bias(t_in, 'conv', layer_idx)
        t_out = act(t_out)
        return t_out

    with torch.no_grad():
        # conv layer 0 [bs, 128, 128, 32]
        layer_idx = 0
        l0 = conv2d(img, 32, layer_idx=layer_idx, eq_lr=eq_lr)
        l0 = post_conv(l0, layer_idx=layer_idx)

        # conv layer 1 [bs, 64, 64, 64]
        layer_idx = 1
        l1 = downsample(l0, factor=2)
        l1 = conv2d(l1, 64, layer_idx=layer_idx, eq_lr=eq_lr)
        l1 = post_conv(l1, layer_idx=layer_idx)

        # conv layer 2 [bs, 32, 32, 128]
        layer_idx = 2
        l2 = downsample(l1, factor=2)
        l2 = conv2d(l2, 128, layer_idx=layer_idx, eq_lr=eq_lr)
        l2 = post_conv(l2, layer_idx=layer_idx)

        # conv layer 3 [bs, 16, 16, 256]
        layer_idx = 3
        l3 = downsample(l2, factor=2)
        l3 = conv2d(l3, 256, layer_idx=layer_idx, eq_lr=eq_lr)
        l3 = post_conv(l3, layer_idx=layer_idx)

        # conv layer 4 [bs, 8, 8, 256]
        layer_idx = 4
        l4 = downsample(l3, factor=2)
        l4 = conv2d(l4, 256, layer_idx=layer_idx, eq_lr=eq_lr)
        l4 = post_conv(l4, layer_idx=layer_idx)

        # conv layer 5 [bs, 4, 4, 512]
        layer_idx = 5
        l5 = downsample(l4, factor=2)
        l5 = conv2d(l5, 512, layer_idx=layer_idx, eq_lr=eq_lr)
        l5 = post_conv(l5, layer_idx=layer_idx)

        # conv layer 6 [bs, 2, 2, 512]
        layer_idx = 6
        l6 = downsample(l5, factor=2)
        l6 = conv2d(l6, 512, layer_idx=layer_idx, eq_lr=eq_lr)
        l6 = post_conv(l6, layer_idx=layer_idx)

        # dense layer 7 [bs, 512]
        layer_idx = 7
        l7 = downsample(l6, factor=2)
        l7 = torch.reshape(l7, [-1, 512])
        l7 = fc(l7, 512, layer_idx=layer_idx, eq_lr=eq_lr)
        l7 = _bias(l7, 'fc', layer_idx=layer_idx, conv=False)
        l7 = act(l7)

        # output layer (dense linear)
        layer_idx = 8
        logits = fc(l7, 1, layer_idx=layer_idx, gain=1., eq_lr=eq_lr)
        logits = _bias(logits, 'fc', layer_idx=layer_idx, conv=False)

    return logits, [l0, l1, l2, l3, l4, l5, l6]
    
"""

def sampler_single(noise_cube, coords, transformations, img_size=128, img_h=None, img_w=None,
                           act=torch.nn.LeakyReLU, scope='sampler', hidden_dim=128, eq_lr=True):
   
    def _add_noise(t_in, noise, layer_idx):
        """
        Add noise in 4 layers. In each layer, add 4 (n_octaves=16) or 8 (n_octaves=32) octaves
        :param t_in: pre-activated layer output
        :param noise: noise octaves
        :param layer_idx: layer index
        :return: pre-activated layer output with added noise octaves
        """

        t_out = t_in
        if n_octaves == 16:
            per_octave_w = torch.nn.init.normal_(torch.empty(4, hidden_dim), 0, 1)
            for i in range(4):
                t_out = t_out + noise[:, :, :, layer_idx * 4 + i:layer_idx * 4 + i + 1] * \
                        torch.reshape(per_octave_w[i, :], [1, 1, 1, hidden_dim])
        elif n_octaves == 32:
            per_octave_w = torch.nn.init.normal_(torch.empty(8, hidden_dim), 0, 1)
            for i in range(8):
                t_out = t_out + noise[:, :, :, layer_idx * 8 + i:layer_idx * 8 + i + 1] * \
                        torch.reshape(per_octave_w[i, :], [1, 1, 1, hidden_dim])
        else:
            import sys
            sys.exit("oops, need n_octaves either 16 or 32!")

        return t_out
    
    def post_conv(t_in, noise, layer_idx=-1):
        """
        Adds octave noise and bias, applies activation function
        """
        t_out = _add_noise(t_in, noise, layer_idx)
        t_out = _bias(t_out, 'conv', layer_idx)
        t_out = act(t_out)

        return t_out
    
    if img_h is None:
        img_h = img_size
    if img_w is None:
        img_w = img_size
    
    n_octaves = noise_cube.shape[0]

    # initialize octaves to 2**i for n_octaves = 16 and 2**(i/2) otherwise
    oct_factors = np.zeros((n_octaves, 3, 3), dtype=np.float32)
    for i in range(n_octaves):
        for j in range(3):
            if n_octaves == 16:
                oct_factors[i, j, j] = 2 ** i
            else:
                oct_factors[i, j, j] = 2 ** (i/2.)
    
    oct_factors = torch.from_numpy(oct_factors)
    transformations = torch.matmul(transformations, oct_factors)

    # obtain octave sampling coordinates by transforming slice coordinates with octave matrices
    # broadcast (same input coords foreach octave)
    coords = torch.unsqueeze(coords, 1)
    coords = coords.repeat(1, n_octaves, 1, 1)
    # now we have a distinct set of coordinates foreach batch and octave
    octave_coords = torch.matmul(transformations, coords)

    # sample noise
    sampled_noise = sample_trilinear(noise_cube, octave_coords, img_h=img_h, img_w=img_w)
    # noise in last dimension (NHWC)
    sampled_noise = sampled_noise.permute(0, 2, 3, 1)

    with torch.no_grad():
        # start with a learned constant
        base = torch.nn.init.constant_(torch.empty(1, 1, 1, hidden_dim), 1)
        # broadcast (constant per pixel and per sample)
        bs = coords.shape[0]
        base = base.repeat(bs, img_h, img_w, 1)

        layer_idx = 0
        l0 = conv2d(base, hidden_dim, layer_idx=layer_idx, kernel_size=1, eq_lr=eq_lr)
        l0 = post_conv(l0, sampled_noise, layer_idx=layer_idx)

        layer_idx = 1
        l1 = conv2d(l0, hidden_dim, layer_idx=layer_idx, kernel_size=1, eq_lr=eq_lr)
        l1 = post_conv(l1, sampled_noise, layer_idx=layer_idx)

        layer_idx = 2
        l2 = conv2d(l1, hidden_dim, layer_idx=layer_idx, kernel_size=1, eq_lr=eq_lr)
        l2 = post_conv(l2, sampled_noise, layer_idx=layer_idx)

        layer_idx = 3
        l3 = conv2d(l2, hidden_dim, layer_idx=layer_idx, kernel_size=1, eq_lr=eq_lr)
        l3 = post_conv(l3, sampled_noise, layer_idx=layer_idx)

        # no noise in final hidden layer
        layer_idx = 4
        l4 = conv2d(l3, hidden_dim, layer_idx=layer_idx, kernel_size=1,  eq_lr=eq_lr)
        l4 = _bias(l4, 'conv', layer_idx)
        l4 = act(l4)

        # 2rgb linear
        layer_idx = 5
        rgb = conv2d(l4, 3, layer_idx=layer_idx, kernel_size=1, gain=1., eq_lr=eq_lr)
        rgb = _bias(rgb, 'conv', layer_idx)

        return rgb
