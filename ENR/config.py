ENR_CONFIG = dict(img_shape=[3, 64, 64], channels_2d=[64, 64, 128, 128, 128, 128, 256, 256, 128, 128, 128],
                  strides_2d=[1, 1, 2, 1, 2, 1, 2, 1, -2, 1, 1], channels_3d=[32, 32, 128, 128, 128, 64, 64, 64],
                  strides_3d=[1, 1, 2, 1, 1, -2, 1, 1], num_channels_inv_projection=[256, 512, 1024],
                  num_channels_projection=[512, 256, 256])

ENR_CONFIG = dict(img_shape=[3, 64, 64], channels_2d=[64, 64, 128, 128, 128, 128, 256, 256, 128, 128, 128],
                  strides_2d=[1, 1, 2, 1, 2, 1, 2, 1, -2, 1, 1], channels_3d=[32, 32, 128, 128, 128, 64, 64, 1],
                  strides_3d=[1, 1, 2, 1, 1, -2, 2, 2], num_channels_inv_projection=[256, 512, 1024],
                  num_channels_projection=[512, 256, 256])


def get_enr_config(final_shape=4, n_channels=1):
    base_dict = dict(img_shape=[3, 64, 64], channels_2d=[64, 64, 128, 128, 128, 128, 256, 256, 128, 128, 128],
                     strides_2d=[1, 1, 2, 1, 2, 1, 2, 1, -2, 1, 1],
                     strides_3d=[1, 1, 2, 1, 1, -2, 1, 1],
                     channels_3d=[32, 32, 128, 128, 128, n_channels, n_channels, n_channels],
                     num_channels_inv_projection=[final_shape // 4, final_shape // 2, final_shape * n_channels],
                     num_channels_projection=[512, 256, 256])
    if final_shape == 4:

        base_dict['strides_2d'] = [1, 1, 2, 1, 2, 1, 2, 1, -2, 2, 2]

    elif final_shape == 16:
        # base_dict['strides_3d'] = [1, 1, 2, 1, 1, -2, 1, 1]
        base_dict['strides_2d'] = [1, 1, 2, 1, 2, 1, 2, 1, -2, 1, 1]
    elif final_shape == 8:
        # base_dict['strides_3d'] = [1, 1, 2, 1, 1, -2, 2, 1]
        base_dict['strides_2d'] = [1, 1, 2, 1, 2, 1, 2, 1, -2, 2, 1]
    return base_dict
