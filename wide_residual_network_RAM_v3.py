from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

def res_adapt_mod(input, dims):
    init=input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Convolution2D(dims, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)
    x = Add()([x, init])

    return x   
    
    
def initial_conv(input):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    x = Convolution2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(input)
    x = res_adapt_mod(x, 16)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    return x


def expand_conv(init, base, k, strides=(1, 1)):
    x = Convolution2D(base * k, (3, 3), padding='same', strides=strides, kernel_initializer='he_normal',
                      use_bias=False)(init)
    x = res_adapt_mod(x, base*k)
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = Convolution2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)
    x = res_adapt_mod(x, base*k)
    
    #new addition v2
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    skip = Convolution2D(base * k, (1, 1), padding='same', strides=strides, kernel_initializer='he_normal',
                      use_bias=False)(init)

    x = Add()([x, skip])
    
    #new addition v2
    x = Activation('relu')(x)
    return x


def conv1_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = Convolution2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(init)
    x = res_adapt_mod(x, 16*k)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    
    #if dropout > 0.0: x = Dropout(dropout)(x)
    x = Convolution2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)
    x = res_adapt_mod(x, 16*k)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x) 

    x = Add()([init, x])
    #switched over in v2
    x = Activation('relu')(x)   
    return x

def conv2_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = Convolution2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(init)
    x = res_adapt_mod(x, 32*k)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    #if dropout > 0.0: x = Dropout(dropout)(x)
    x = Convolution2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)
    x = res_adapt_mod(x, 32*k)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)

    x = Add()([init, x])
    x = Activation('relu')(x)
    return x

def conv3_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = Convolution2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(init)
    x = res_adapt_mod(x,64*k)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    #if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)
    x = res_adapt_mod(x, 64*k)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    
    
    x = Add()([init, x])
    x = Activation('relu')(x)

    return x

def create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1):
    """
    Creates a Wide Residual Network with specified parameters

    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param verbose: Debug info to describe created WRN
    :return:
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    ip = Input(shape=input_dim)

    x = initial_conv(ip)
    nb_conv = 4

    x = expand_conv(x, 16, k)

    for i in range(N - 1):
        x = conv1_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 32, k, strides=(2, 2))

    for i in range(N - 1):
        x = conv2_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 64, k, strides=(2, 2))

    for i in range(N - 1):
        x = conv3_block(x, k, dropout)
        nb_conv += 2
        
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(ip, x)

    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model

if __name__ == "__main__":
    from keras.utils import plot_model
    from keras.layers import Input
    from keras.models import Model

    init = (32, 32, 3)

    wrn_28_10 = create_wide_residual_network(init, nb_classes=10, N=2, k=2, dropout=0.0)

    wrn_28_10.summary()

    plot_model(wrn_28_10, "WRN-16-2.png", show_shapes=True, show_layer_names=True)
