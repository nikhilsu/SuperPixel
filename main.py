from gan_model.conditional_gan import CGAN
from gan_model.config import Config
from gan_model.discriminator import Discriminator
from gan_model.generator import Generator
import numpy as np


def train():
    generator, discriminator = Generator(), Discriminator()
    gan = CGAN(generator, discriminator)
    gan.compile_model()
    dataset = np.load('dataset/100.npy')

    # Normalize input
    dataset = (dataset - 127.5) / 127.5

    real = np.ones((Config.batch_size(), 1))
    fake = np.zeros((Config.batch_size(), 1))

    for epoch in range(Config.epochs()):
        np.random.shuffle(dataset)
        batch = dataset[:Config.batch_size()]
        batch = [(low, high) for low, high in batch]
        low_res, high_res = [np.array(tup) for tup in zip(*batch)]
        del batch[:]

        generated_images = generator.predict(low_res)
        print('Shape: {}'.format(str(generated_images.shape)))
        valid_loss = discriminator.train_on_batch(high_res, real)
        fake_loss = discriminator.train_on_batch(generated_images, fake)
        d_loss = 0.5 * np.add(valid_loss, fake_loss)

        g_loss = gan.train_on_batch(low_res, real)
        loss = d_loss[0]
        acc = 100 * d_loss[1]
        print('{} -> Disc loss: {:0.8f}, acc.: {:0.2f}, Gen loss: {:0.8f}'.format(epoch, loss, acc, g_loss))


if __name__ == '__main__':
    train()
