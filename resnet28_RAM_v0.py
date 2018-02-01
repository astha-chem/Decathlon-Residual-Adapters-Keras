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

def pre_layers_conv(input, filters = 32, factor=1, learnall = True):
    x = Convolution2D(filters*factor, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False, trainable = learnall)(input)
    x = res_adapt_mod(x, filters*factor)
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    return x


def conv_block(input, filters=64, factor=1, learnall = True):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    #conv1
    x = Convolution2D(filters*factor, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False, trainable = learnall)(input)
    x = res_adapt_mod(x, filters*factor)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    #conv2
    x = Convolution2D(filters*factor, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False, trainable = learnall)(x)
    x = res_adapt_mod(x, filters*factor)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
   
    #summing up
    m = Add()([init, x])
    
    x = Activation('relu')(m)
    return x

def conv_scaledown(init, filters=64, factor=1, strides=(1, 1), learnall = True):
    
    x = Convolution2D(filters*factor, (3, 3), padding='same', strides=strides, kernel_initializer='he_normal',
                      use_bias=False, trainable = learnall)(init)
    x = res_adapt_mod(x, filters*factor)
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = Convolution2D(filters*factor, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False, trainable = learnall)(x)
    x = res_adapt_mod(x, filters*factor)
    
    #new addition v2
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    
    skip = Convolution2D(filters*factor, (1, 1), padding='same', strides=strides, kernel_initializer='he_normal',
                      use_bias=False)(init)
    x = Add()([skip, x])
    
    #new addition v2
    x = Activation('relu')(x)
    return x

def create_resnet_RAM(input_dim, filters=32, factor=1, nb_classes=100, N=4, verbose=1, learnall = True, name = 'imagenet12'):
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

    x = pre_layers_conv(ip, filters = 32, factor=1,  learnall = learnall)
    nb_conv = 1
    x = conv_scaledown(x, filters=64, factor=1, strides = (2,2), learnall = learnall)    
    
    for i in range(N-1):
        x = conv_block(x, filters=64, factor=1, learnall = learnall)
        nb_conv += 2
    x = conv_scaledown(x, filters=128, factor=1, strides = (2,2), learnall = learnall)
    for i in range(N-1):
        x = conv_block(x, filters=128, factor=1, learnall = learnall)
        nb_conv += 2
    x = conv_scaledown(x, filters=256, factor=1, strides = (2,2), learnall = learnall)   
    for i in range(N-1):
        x = conv_block(x, filters=256, factor=1, learnall = learnall)
        nb_conv += 2    
    nb_conv+= 6    
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    
    x = AveragePooling2D((4,4))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, activation='softmax', name = name)(x)

    model = Model(ip, x)

    if verbose: print("ResNet-%d-%d with RAM created." % (nb_conv, factor))
    return model

if __name__ == "__main__":
    from keras.utils import plot_model
    from keras.layers import Input
    from keras.models import Model

    init = (32, 32, 3)

    wrn_28_10 = create_wide_residual_network(init, nb_classes=10, N=2, k=2, dropout=0.0)

    wrn_28_10.summary()

    plot_model(wrn_28_10, "WRN-16-2.png", show_shapes=True, show_layer_names=True)
