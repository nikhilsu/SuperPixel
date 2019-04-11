from gan_model.conditional_gan import CGAN
from gan_model.config import Config
from gan_model.discriminator import Discriminator
from gan_model.generator import Generator
from tqdm import tqdm

import numpy as np
import os
import cv2


def normalize_image(image):
    return (image - 127.5) / 127.5


def de_normalize_image(image):
    return 127.5 * image + 127.5


def flush_image_pair_to_disk(generator, image_pairs, i):
    low_res, high_res = image_pairs
    generated_images = generator.predict(low_res)

    # Rescale image
    generated_images = de_normalize_image(generated_images)
    low_res = de_normalize_image(low_res)
    high_res = de_normalize_image(high_res)
    os.makedirs(Config.checkpoint_output(), exist_ok=True)
    checkpoint_dir = os.path.join(Config.checkpoint_output(), str(i))
    os.makedirs(checkpoint_dir)

    for i in range(len(low_res)):
        input_image = low_res[i].astype(int)
        generated_image = generated_images[i].astype(int)
        ground_truth = high_res[i].astype(int)
        cv2.imwrite(os.path.join(checkpoint_dir, 'input_{}.jpg'.format(i)), input_image)
        cv2.imwrite(os.path.join(checkpoint_dir, 'generated_{}.jpg'.format(i)), generated_image)
        cv2.imwrite(os.path.join(checkpoint_dir, 'ground_truth_{}.jpg'.format(i)), ground_truth)


def train():
    generator, discriminator = Generator(), Discriminator()
    gan = CGAN(generator, discriminator)
    gan.compile_model()
    dataset = np.load('dataset/100.npy')

    # Normalize input
    dataset = normalize_image(dataset)

    real = np.ones((Config.batch_size(), 1))
    fake = np.zeros((Config.batch_size(), 1))

    for epoch in tqdm(range(Config.epochs()), desc='Training Epochs'):
        np.random.shuffle(dataset)
        batch = dataset[:Config.batch_size()]
        batch = [(low, high) for low, high in batch]
        low_res, high_res = [np.array(tup) for tup in zip(*batch)]
        del batch[:]

        generated_images = generator.predict(low_res)
        valid_loss = discriminator.train_on_batch(high_res, real)
        fake_loss = discriminator.train_on_batch(generated_images, fake)
        d_loss = 0.5 * np.add(valid_loss, fake_loss)

        g_loss = gan.train_on_batch(low_res, real)
        loss = d_loss[0]
        acc = 100 * d_loss[1]
        print('{} -> Disc loss: {:0.8f}, acc.: {:0.2f}, Gen loss: {:0.8f}'.format(epoch, loss, acc, g_loss))

        if Config.checkpoint_reached(epoch):
            random_samples = np.random.randint(0, Config.batch_size(), Config.checkpoint_batch_size())
            flush_image_pair_to_disk(generator, [low_res[random_samples], high_res[random_samples]], epoch)


if __name__ == '__main__':
    train()
