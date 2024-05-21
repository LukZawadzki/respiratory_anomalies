import tensorflow as tf
from keras import layers


def create_vgg(dense_neurons=256, input_shape=(224, 224, 3), num_classes=4):
    """
    Creates a model with VGG backbone
    :param dense_neurons: number of units in dense layers
    :param input_shape: shape of input image
    :param num_classes: number of classes
    :return: model
    """
    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    vgg.trainable = False

    flatten = layers.Flatten()(vgg.output)
    dense = layers.Dense(dense_neurons, activation='relu')(flatten)
    dense = layers.Dense(dense_neurons, activation='relu')(dense)
    output = layers.Dense(num_classes, activation='softmax')(dense)

    model = tf.keras.Model(inputs=vgg.input, outputs=output)

    model.summary()

    return model
