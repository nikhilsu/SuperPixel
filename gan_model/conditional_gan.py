from keras import Input, Model

from gan_model.config import Config
from gan_model.gancomponent import GANComponent


class CGAN(GANComponent):
    def __init__(self, generator, discriminator):
        discriminator.compile_model()
        generator_input_tensor = Input(shape=(Config.image_shape()))
        generator_output_tensor = generator.setup_input_tensor(generator_input_tensor)

        discriminator_output = discriminator.setup_input_tensor(generator_output_tensor)

        super().__init__(Model(generator_input_tensor, discriminator_output))

    def compile_model(self):
        super()._compile()
