import dataloaders
import logging
import matplotlib.pyplot as plt
import numpy as np
import models.unetModel
import models.vggModel


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.WARNING)

    ds_train, ds_test = dataloaders.DataSetLoader('temp/ICBHI_final_database', 44100, 0, 20).create_dataset(16, 0.8)

    # model = models.unetModel.create_model_smaller(input_shape=(128, 512, 1), dense_top=True)
    # model = models.unetModel.create_model_big(input_shape=(128, 512, 1), dense_top=True)
    # model = models.vggModel.create_vgg(input_shape=(128, 512, 1))

    # Kompilacja modelu
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

    # Trenowanie modelu
    history = model.fit(ds_train, epochs=10)

    # Wyświetlenie wyników treningu
    print("Historia treningu:", history.history)