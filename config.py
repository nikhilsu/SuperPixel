from keras.optimizers import Adam


class Config:
    @staticmethod
    def low_res_img_dims():
        return 28, 28

    @staticmethod
    def high_res_img_dims():
        return 28, 28

    @staticmethod
    def channels():
        return 3

    @staticmethod
    def high_res_img_shape():
        dims = Config.high_res_img_dims()
        return dims[0], dims[1], Config.channels()

    @staticmethod
    def adam_optimizer():
        return Adam(0.0002, 0.5)
