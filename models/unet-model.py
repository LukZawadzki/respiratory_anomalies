import keras
from keras import layers


def create_model_big(dense_neurons=256, input_shape=(224, 224, 3), num_classes=4):
    """
    Classic UNet model, as per https://arxiv.org/abs/1505.04597
    :param dense_neurons: number of units in dense layers
    :param input_shape: shape of input image
    :param num_classes: number of classes
    :return: model
    """

    # Downscaling part of the UNet
    input = layers.Input(shape=input_shape)
    conv2d_1 = layers.Conv2D(64, 3, activation='relu', padding='same')(input)
    conv2d_1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2d_1)

    pool2d = layers.MaxPooling2D(pool_size=(2, 2))(conv2d_1)
    conv2d_2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2d)
    conv2d_2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2d_2)

    pool2d = layers.MaxPooling2D(pool_size=(2, 2))(conv2d_2)
    conv2d_3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2d)
    conv2d_3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv2d_3)

    pool2d = layers.MaxPooling2D(pool_size=(2, 2))(conv2d_3)
    conv2d_4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool2d)
    conv2d_4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv2d_4)

    pool2d = layers.MaxPooling2D(pool_size=(2, 2))(conv2d_4)
    conv2d_5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool2d)
    conv2d_5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv2d_5)

    # Upscaling part of the UNet
    upscale_1 = layers.UpSampling2D(size=(2, 2))(conv2d_5)
    concat_1 = layers.concatenate([conv2d_4, upscale_1], axis=-1)
    conv2d_6 = layers.Conv2D(512, 3, activation='relu', padding='same')(concat_1)
    conv2d_6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv2d_6)

    upscale_2 = layers.UpSampling2D(size=(2, 2))(conv2d_6)
    concat_2 = layers.concatenate([conv2d_3, upscale_2], axis=-1)
    conv2d_7 = layers.Conv2D(256, 3, activation='relu', padding='same')(concat_2)
    conv2d_7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv2d_7)

    upscale_3 = layers.UpSampling2D(size=(2, 2))(conv2d_7)
    concat_3 = layers.concatenate([conv2d_2, upscale_3], axis=-1)
    conv2d_8 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat_3)
    conv2d_8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2d_8)

    upscale_4 = layers.UpSampling2D(size=(2, 2))(conv2d_8)
    concat_4 = layers.concatenate([conv2d_1, upscale_4], axis=-1)
    conv2d_9 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat_4)
    conv2d_9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2d_9)

    dense = layers.Dense(dense_neurons, activation='relu')(conv2d_9)
    dense = layers.Dense(dense_neurons, activation='relu')(dense)
    output = layers.Dense(num_classes, activation='softmax')(dense)

    model = keras.models.Model(inputs=input, outputs=output)
    model.summary()

    return model


def create_model_smaller(dense_neurons=256, input_shape=(224, 224, 3), num_classes=4):
    """
    Smaller variant of the UNet model - with just three skip-blocks, instead of four
    :param dense_neurons: number of units in dense layers
    :param input_shape: shape of input image
    :param num_classes: number of classes
    :return: model
    """
    # Downscaling part of the UNet
    input = layers.Input(shape=input_shape)
    conv2d_1 = layers.Conv2D(64, 3, activation='relu', padding='same')(input)
    conv2d_1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2d_1)

    pool2d = layers.MaxPooling2D(pool_size=(2, 2))(conv2d_1)
    conv2d_2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2d)
    conv2d_2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2d_2)

    pool2d = layers.MaxPooling2D(pool_size=(2, 2))(conv2d_2)
    conv2d_3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2d)
    conv2d_3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv2d_3)

    pool2d = layers.MaxPooling2D(pool_size=(2, 2))(conv2d_3)
    conv2d_4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool2d)
    conv2d_4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv2d_4)

    # Upscaling part of the UNet
    upscale_2 = layers.UpSampling2D(size=(2, 2))(conv2d_4)
    concat_2 = layers.concatenate([conv2d_3, upscale_2], axis=-1)
    conv2d_7 = layers.Conv2D(256, 3, activation='relu', padding='same')(concat_2)
    conv2d_7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv2d_7)

    upscale_3 = layers.UpSampling2D(size=(2, 2))(conv2d_7)
    concat_3 = layers.concatenate([conv2d_2, upscale_3], axis=-1)
    conv2d_8 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat_3)
    conv2d_8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2d_8)

    upscale_4 = layers.UpSampling2D(size=(2, 2))(conv2d_8)
    concat_4 = layers.concatenate([conv2d_1, upscale_4], axis=-1)
    conv2d_9 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat_4)
    conv2d_9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2d_9)

    dense = layers.Dense(dense_neurons, activation='relu')(conv2d_9)
    dense = layers.Dense(dense_neurons, activation='relu')(dense)
    output = layers.Dense(num_classes, activation='softmax')(dense)

    model = keras.models.Model(inputs=input, outputs=output)
    model.summary()

    return model
