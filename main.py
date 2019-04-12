from gan_model.conditional_gan import CGAN
from gan_model.config import Config
from gan_model.discriminator import Discriminator
from gan_model.generator import Generator
from tqdm import tqdm
from matplotlib import pyplot as plt

import numpy as np
import os
import cv2
import sys


def normalize_image(image):
    return (image - 127.5) / 127.5


def de_normalize_image(image):
    return 127.5 * image + 127.5


def flush_image_pair_to_disk(generator, image_pairs, e):
    low_res, high_res = image_pairs
    generated_images = generator.predict(low_res)

    generated_images = de_normalize_image(generated_images)
    low_res = de_normalize_image(low_res)
    high_res = de_normalize_image(high_res)
    os.makedirs(Config.checkpoint_output(), exist_ok=True)
    checkpoint_dir = os.path.join(Config.checkpoint_output(), str(e))
    os.makedirs(checkpoint_dir, exist_ok=True)

    for j in range(len(low_res)):
        image_triplet = [low_res[j].astype('uint8'), generated_images[j].astype('uint8'), high_res[j].astype('uint8')]
        names = ['Low Resolution', 'Generated Image', 'Ground Truth']

        fig, axes = plt.subplots(nrows=1, ncols=3)
        for k, axis in enumerate(axes):
            axis.imshow(cv2.cvtColor(image_triplet[k], cv2.COLOR_BGR2RGB))
            axis.set_title(names[k])
            axis.axis('off')

        fig.savefig(os.path.join(checkpoint_dir, 'checkpoint_{}.jpg'.format(j)))
        plt.close()


def train(dataset_path):
    generator, discriminator = Generator(), Discriminator()
    gan = CGAN(generator, discriminator)
    gan.compile_model()
    dataset = np.load(dataset_path)

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
    if ('LOCAL' not in os.environ) and len(sys.argv) != 2:
        print('Usage: python prepare_dataset.py <path_to_dataset>')
        sys.exit(1)

    path = sys.argv[1] if len(sys.argv) == 2 else 'dataset/100.npy'
    train(path)
