import dataloaders
import logging
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.WARNING)

    ds_train, ds_test = dataloaders.DataSetLoader('temp/ICBHI_final_database', 44100, 0, 20).create_dataset(32, 0.8)

    for mfcc, label in ds_train.take(1):
        print(mfcc.shape)
        print(label.shape)

        example_mfcc = mfcc[0].numpy()

        plt.imshow(example_mfcc, cmap='hot')

        plt.show()

