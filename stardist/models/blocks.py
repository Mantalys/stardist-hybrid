from csbdeep.utils.tf import keras_import, BACKEND as K
GlobalAveragePooling2D, DepthwiseConv2D, LayerNormalization, Reshape, Dense, Multiply, Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D, Cropping2D, Cropping3D, Concatenate, Add, Dropout, Activation, BatchNormalization, SeparableConv2D = \
    keras_import('layers', "GlobalAveragePooling2D", "DepthwiseConv2D", "LayerNormalization", "Reshape", "Dense", "Multiply", 'Conv2D', 'MaxPooling2D', 'UpSampling2D', 'Conv3D', 'MaxPooling3D', 'UpSampling3D', 'Cropping2D', 'Cropping3D', 'Concatenate', 'Add', 'Dropout', 'Activation', 'BatchNormalization', "SeparableConv2D")

from csbdeep.utils import backend_channels_last

def se_module(x, reduction=16, name=None):
    """Squeeze-and-Excitation channel attention."""
    filters = K.int_shape(x)[-1]
    se = GlobalAveragePooling2D(name=None if name is None else f"{name}_gap")(x)
    se = Reshape((1,1,filters))(se)
    se = Dense(filters // reduction, activation="relu", use_bias=False,
               kernel_initializer="he_normal")(se)
    se = Dense(filters, activation="sigmoid", use_bias=False,
               kernel_initializer="he_normal")(se)
    return Multiply(name=None if name is None else f"{name}_scale")([x, se])


def vanilla_block(n_filter, kernel_size=(3,3), activation="relu",
                  batch_norm=False, kernel_init="glorot_uniform", name=None):
    def f(x):
        for i in range(2):
            x = Conv2D(n_filter, kernel_size, padding="same",
                       kernel_initializer=kernel_init,
                       use_bias=not batch_norm,
                       name=None if name is None else f"{name}_conv{i}")(x)
            if batch_norm: x = BatchNormalization()(x)
            x = Activation(activation)(x)
        return x
    return f


def residual_block(n_filter, kernel_size=(3,3), activation="relu",
                   batch_norm=False, kernel_init="he_normal", name=None):
    pool = (1,1)
    def f(inp):
        x = Conv2D(n_filter, kernel_size, strides=pool,
                   padding="same", kernel_initializer=kernel_init,
                   use_bias=not batch_norm)(inp)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(activation)(x)

        x = Conv2D(n_filter, kernel_size, padding="same",
                   kernel_initializer=kernel_init,
                   use_bias=not batch_norm)(x)
        if batch_norm: x = BatchNormalization()(x)

        if any(p!=1 for p in pool) or n_filter != K.int_shape(inp)[-1]:
            inp = Conv2D(n_filter, (1,1), strides=pool,
                         padding="same", kernel_initializer=kernel_init)(inp)
        x = Add()([inp, x])
        x = Activation(activation)(x)
        return x
    return f


def sepconv_block(n_filter, kernel_size=(3,3), activation="relu",
                  batch_norm=False, kernel_init="glorot_uniform", name=None):
    def f(x):
        for i in range(2):
            x = SeparableConv2D(n_filter, kernel_size, padding="same",
                                depthwise_initializer=kernel_init,
                                pointwise_initializer=kernel_init,
                                use_bias=not batch_norm)(x)
            if batch_norm: x = BatchNormalization()(x)
            x = Activation(activation)(x)
        return x
    return f


def inverted_res_block(n_filter, kernel_size=(3,3), activation="relu",
                       batch_norm=False, kernel_init="he_normal", name=None):
    expansion = 4
    pool = (1,1)
    def f(inp):
        mid = n_filter * expansion
        x = Conv2D(mid, (1,1), padding="same", kernel_initializer=kernel_init,
                   use_bias=not batch_norm)(inp)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(activation)(x)

        x = DepthwiseConv2D(kernel_size, strides=pool, padding="same",
                            depthwise_initializer=kernel_init,
                            use_bias=not batch_norm)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(activation)(x)

        x = Conv2D(n_filter, (1,1), padding="same", kernel_initializer=kernel_init,
                   use_bias=not batch_norm)(x)
        if batch_norm: x = BatchNormalization()(x)

        if pool==(1,1) and K.int_shape(inp)[-1]==n_filter:
            x = Add()([inp, x])
        return x
    return f


def convnext_block(n_filter, kernel_size=7, activation="gelu",
                   batch_norm=False, kernel_init="he_normal", name=None):
    expansion = 4
    def f(inp):
        x = DepthwiseConv2D(kernel_size, padding="same",
                            depthwise_initializer=kernel_init)(inp)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Conv2D(n_filter*expansion, 1, kernel_initializer=kernel_init)(x)
        x = Activation(activation)(x)
        x = Conv2D(n_filter, 1, kernel_initializer=kernel_init)(x)
        x = Add()([inp, x])
        return x
    return f


BLOCKS = {
    "vanilla": vanilla_block,
    "residual": residual_block,
    "sepconv": sepconv_block,
    "invertedres": inverted_res_block,
    "convnext": convnext_block,
}

def block_factory(block_type, n_filter, kernel_size=(3,3), activation="relu", 
                  batch_norm=False, kernel_init="glorot_uniform", use_se=False, se_reduction=16,
                  name=None, **kwargs):
    block_fn = BLOCKS[block_type](n_filter, kernel_size=kernel_size,
                                  activation=activation, batch_norm=batch_norm,
                                  kernel_init=kernel_init,
                                  name=name, **kwargs)
    def wrapped(inp):
        x = block_fn(inp)
        if use_se:
            x = se_module(x, reduction=se_reduction, name=name)
        return x
    return wrapped


def unet_block(n_depth=2, n_filter_base=16, kernel_size=(3,3), n_conv_per_depth=2,
               activation="relu",
               batch_norm=False,
               dropout=0.0,
               last_activation=None,
               pool=(2,2),
               kernel_init="glorot_uniform",
               expansion=2,
               prefix='',
               block_type="vanilla",
               use_se=False):

    if len(pool) != len(kernel_size):
        raise ValueError('kernel and pool sizes must match.')
    n_dim = len(kernel_size)
    if n_dim not in (2,3):
        raise ValueError('unet_block only 2d or 3d.')

    pooling    = MaxPooling2D if n_dim == 2 else MaxPooling3D
    upsampling = UpSampling2D if n_dim == 2 else UpSampling3D

    if last_activation is None:
        last_activation = activation

    channel_axis = -1 if backend_channels_last() else 1

    def _name(s):
        return prefix+s
    
    def _func(input):
        skip_layers = []
        layer = input

        # down path
        for n in range(n_depth):
            for i in range(n_conv_per_depth):
                layer = block_factory(
                    block_type=block_type,
                    n_filter=int(n_filter_base * expansion ** n),
                    kernel_size=kernel_size,
                    activation=activation,
                    batch_norm=batch_norm,
                    kernel_init=kernel_init,
                    use_se=use_se,
                    name=_name(f"down_level_{n}_no_{i}")
                )(layer)
            print(layer)
            skip_layers.append(layer)
            layer = pooling(pool, name=_name(f"max_{n}"))(layer)

        # middle path
        for i in range(n_conv_per_depth - 1):
            layer = block_factory(
                block_type=block_type,
                n_filter=int(n_filter_base * expansion ** n_depth),
                kernel_size=kernel_size,
                activation=activation,
                batch_norm=batch_norm,
                kernel_init=kernel_init,
                use_se=use_se,
                name=_name(f"middle_{i}")
            )(layer)

        layer = block_factory(
            block_type=block_type,
            n_filter=int(n_filter_base * expansion ** max(0, n_depth - 1)),
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            kernel_init=kernel_init,
            use_se=use_se,
            name=_name(f"middle_{n_conv_per_depth}")
        )(layer)

        # up path with skip connections
        for n in reversed(range(n_depth)):
            layer = Concatenate(axis=channel_axis)([upsampling(pool)(layer), skip_layers[n]])
            for i in range(n_conv_per_depth - 1):
                layer = block_factory(
                    block_type=block_type,
                    n_filter=int(n_filter_base * expansion ** n),
                    kernel_size=kernel_size,
                    activation=activation,
                    batch_norm=batch_norm,
                    kernel_init=kernel_init,
                    use_se=use_se,
                    name=_name(f"up_level_{n}_no_{i}")
                )(layer)

            layer = block_factory(
                block_type=block_type,
                n_filter=int(n_filter_base * expansion ** max(0, n - 1)),
                kernel_size=kernel_size,
                activation=activation if n > 0 else last_activation,
                batch_norm=batch_norm,
                kernel_init=kernel_init,
                use_se=use_se,
                name=_name(f"up_level_{n}_no_{n_conv_per_depth}")
            )(layer)

        return layer
    return _func
