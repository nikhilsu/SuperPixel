from keras import Input, Model

from config import Config


class CGAN(object):
    def __init__(self, generator, discriminator):
        discriminator.compile()
        generator_input_tensor = Input(shape=(Config.low_res_img_dims()))
        generator_output_tensor = generator.setup_input_tensor(generator_input_tensor)

        discriminator_output = discriminator.setup_input_tensor(generator_output_tensor)

        self.model = Model(generator_input_tensor, discriminator_output)

    def compile(self):
        self.model.compile(loss='binary_crossentropy', optimizer=Config.optimizer())

    def train_on_batch(self, input_tensor, output_tensor):
        self.model.train_on_batch(input_tensor, output_tensor)
