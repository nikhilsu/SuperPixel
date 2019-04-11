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
        return 1000
