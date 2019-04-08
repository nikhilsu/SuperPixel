from keras import Sequential, Input, Model
from keras.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, Flatten

from config import Config


class Discriminator(object):
    def __init__(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=Config.high_res_img_shape(), padding="same"))
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

        input_img = Input(shape=Config.high_res_img_shape())
        validity = model(input_img)

        self.model = Model(input_img, validity)
        self.model.trainable = False

    def compile(self):
        self.model.compile(loss='binary_crossentropy',
                           optimizer=Config.optimizer(),
                           metrics=['accuracy'])

    def setup_input_tensor(self, tensor):
        return self.model(tensor)

    def train_on_batch(self, input_tensor, output_tensor):
        self.model.train_on_batch(input_tensor, output_tensor)

