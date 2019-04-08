from keras import Sequential, Input, Model
from keras.layers import Dense, Reshape, UpSampling2D, Conv2D, BatchNormalization, Activation

from config import Config


class Generator(object):
    def __init__(self):
        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=Config.low_res_img_dims()))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(Config.channels(), kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        input_img = Input(shape=Config.low_res_img_dims())
        high_res_output_img = model(input_img)

        self.model = Model(input_img, high_res_output_img)

    def compile(self):
        self.model.compile(loss='binary_crossentropy', optimizer=Config.optimizer())

    def setup_input_tensor(self, tensor):
        return self.model(tensor)
