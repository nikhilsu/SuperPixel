from keras import Sequential, Input, Model
from keras.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, Flatten

from config import Config
from gancomponent import GANComponent


class Discriminator(GANComponent):
    def __init__(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=Config.image_shape(), padding="same",
                         data_format="channels_last"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        input_img = Input(shape=Config.image_shape())
        validity = model(input_img)

        self.model = Model(input_img, validity)
        self.model.trainable = False
        super().__init__(model)

    def compile_model(self):
        super()._compile(metrics=['accuracy'])
