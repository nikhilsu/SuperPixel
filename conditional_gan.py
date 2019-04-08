from keras import Input, Model

from config import Config
from gancomponent import GANComponent


class CGAN(GANComponent):
    def __init__(self, generator, discriminator):
        discriminator.compile()
        generator_input_tensor = Input(shape=(Config.low_res_img_dims()))
        generator_output_tensor = generator.setup_input_tensor(generator_input_tensor)

        discriminator_output = discriminator.setup_input_tensor(generator_output_tensor)

        super().__init__(Model(generator_input_tensor, discriminator_output))

    def compile_model(self):
        super().compile()
