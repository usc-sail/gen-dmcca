from keras.layers import *
from keras.models import Sequential, Model
from keras import optimizers
from keras.regularizers import l2
from objectives_mcca import mcca_loss

fsize = 32
kernel_size = 3
pool_size = 2
stride = (2, 2)
input_shape = (1, 98, 64)

def create_layers_small(act_ = 'relu'):
    layers_list =  [
        Conv2D(fsize, kernel_size, padding='same'),
        Conv2D(fsize, kernel_size, padding='same'),
        BatchNormalization(),
        Activation(act_),
        MaxPooling2D(pool_size=(pool_size, pool_size), strides=stride, padding='same'),

        Conv2D(2*fsize, kernel_size, padding='same'),
        Conv2D(2*fsize, kernel_size, padding='same'),
        BatchNormalization(),
        Activation(act_),
        MaxPooling2D(pool_size=(pool_size, pool_size), strides=stride, padding='same'),

        Conv2D(4*fsize, kernel_size, padding='same'),
        Conv2D(4*fsize, kernel_size, padding='same'),
        BatchNormalization(),
        Activation(act_),
        GlobalAveragePooling2D(),
        Dense(128),
        BatchNormalization(),
        Activation(act_),
    ]
    return layers_list

def create_layers(act_ = 'relu'):
    layers_list =  [
        Conv2D(fsize, kernel_size, padding='same'),
        BatchNormalization(),
        Activation(act_),
        Conv2D(fsize, kernel_size, padding='same'),
        BatchNormalization(),
        Activation(act_),
        MaxPooling2D(pool_size=(pool_size, pool_size), strides=stride, padding='same'),

        Conv2D(2*fsize, kernel_size, padding='same'),
        BatchNormalization(),
        Activation(act_),
        Conv2D(2*fsize, kernel_size, padding='same'),
        BatchNormalization(),
        Activation(act_),
        MaxPooling2D(pool_size=(pool_size, pool_size), strides=stride, padding='same'),

        Conv2D(4*fsize, kernel_size, padding='same'),
        BatchNormalization(),
        Activation(act_),
        Conv2D(4*fsize, kernel_size, padding='same'),
        BatchNormalization(),
        Activation(act_),
        Conv2D(4*fsize, kernel_size, padding='same'),
        BatchNormalization(),
        Activation(act_),
        MaxPooling2D(pool_size=(pool_size, pool_size), strides=stride, padding='same'),
        GlobalAveragePooling2D(),
        Dense(64),
        BatchNormalization(),
        Activation(act_),
    ]
    return layers_list

# create a model!
def create_model(act_ = 'relu', n_modalities=3, learning_rate=1e-2, gamma=0.4, reg_par=1e-5):
    """
    Input:
    ..
   Output:
    ..

    builds the whole model form a list of list of layer sizes!
    !!## note this is not the Sequential style model!
    """    
    input_layers = [ Input(input_shape) for i in range(n_modalities) ]

    fc_output_layer_list = []

    for l_i in range(n_modalities):
        # pre-create the dense(fc) layers you need
        ## USING ONLY LINEAR ACTIVATIONS FOR NOW!!
        fc_layers_ = create_layers(act_ = act_)
        D = fc_layers_[0](input_layers[l_i])
        # do this in a non-sequential style Keras model
        for d_i, d in enumerate(fc_layers_[1:]): D = d(D) 
        fc_output_layer_list.append(D)

    output = concatenate(fc_output_layer_list)
    model = Model(input_layers, [output])

    rms_prop = optimizers.RMSprop(lr=learning_rate)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=mcca_loss(n_modalities, 0.2), optimizer=sgd)

    return model

