from keras import Sequential, Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation

from config import Config
from gancomponent import GANComponent


class Generator(GANComponent):
    def __init__(self):
        model = Sequential()

        model.add(Conv2D(128, kernel_size=3, padding="same", input_shape=(128, 128, 3), data_format="channels_last"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(Config.channels(), kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        input_img = Input(shape=Config.image_shape())
        high_res_output_img = model(input_img)

        super().__init__(Model(input_img, high_res_output_img))

    def compile_model(self):
        super()._compile()
