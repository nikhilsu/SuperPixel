from conditional_gan import CGAN
from discriminator import Discriminator
from generator import Generator


def train():
    gen, discriminator = Generator(), Discriminator()
    gan = CGAN(gen, discriminator)
    gan.compile_model()



