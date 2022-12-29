ENR_CONFIG = dict(img_shape=[3, 64, 64], channels_2d=[64, 64, 128, 128, 128, 128, 256, 256, 128, 128, 128],
                       strides_2d=[1, 1, 2, 1, 2, 1, 2, 1, -2, 1, 1], channels_3d=[32, 32, 128, 128, 128, 64, 64, 64],
                       strides_3d=[1, 1, 2, 1, 1, -2, 1, 1], num_channels_inv_projection=[256, 512, 1024],
                       num_channels_projection=[512, 256, 256])