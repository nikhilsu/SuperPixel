import os
from keras.optimizers import Adam


class Config:
    @staticmethod
    def channels():
        return 3

    @staticmethod
    def image_shape():
        return 128, 128, Config.channels()

    @staticmethod
    def adam_optimizer():
        return Adam(0.0002, 0.5)

    @staticmethod
    def batch_size():
        return 32

    @staticmethod
    def epochs():
        small = 'LOCAL' in os.environ
        return 100 if small else 60000

    @staticmethod
    def checkpoint_reached(epoch):
        n = 1 if 'LOCAL' in os.environ else 1000
        return epoch % n == 0

    @staticmethod
    def checkpoint_batch_size():
        return 5

    @staticmethod
    def checkpoint_output():
        return 'checkpoint_images'
