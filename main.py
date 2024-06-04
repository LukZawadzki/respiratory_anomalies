import dataloaders
import logging
import matplotlib.pyplot as plt
import numpy as np
import models.unetModel
import models.vggModel
import tensorflow as tf

# tf.config.run_functions_eagerly(True)


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.WARNING)

    ds_train, ds_test = dataloaders.DataSetLoader('temp/preprocessed').create_dataset(64, 0.8)

    for mfcc, label in ds_train.take(1):
        print(mfcc.shape)
        print(label.shape)

    # model = models.unetModel.create_model_smaller(input_shape=(128, 512, 1), dense_top=True)
    # model = models.unetModel.create_model_big(input_shape=(128, 512, 1), dense_top=True)
    model = models.vggModel.create_vgg(input_shape=(64, 128, 3))

    # Kompilacja modelu
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], run_eagerly=False)

    # Trenowanie modelu
    history = model.fit(ds_train.repeat(10), epochs=10)

    # Wyświetlenie wyników treningu
    print("Historia treningu:", history.history)