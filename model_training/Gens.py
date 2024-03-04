import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import nibabel as nib


def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(128 * 7 * 7, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((7, 7, 128)))
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(1, (7,7), activation='tanh', padding='same'))
    return model


def build_discriminator(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def load_nifti_files(data_dir):
    nii_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".nii"):
                nii_files.append(os.path.join(root, file))
    return nii_files


def read_nifti(filepath):
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return array


data_dir = '../input/liver-tumor-segmentation'
real_images = [read_nifti(file) for file in load_nifti_files(data_dir)]


img_shape = real_images[0].shape
latent_dim = 100


discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


generator = build_generator(latent_dim)


gan = build_gan(generator, discriminator)

# Training loop
def train_gan(generator, discriminator, gan, real_images, latent_dim, n_epochs=100, batch_size=128):
    for epoch in range(n_epochs):
        for i in range(0, len(real_images), batch_size):
            
            real_batch = real_images[i:i+batch_size]
            fake_images = generator.predict(tf.random.normal(shape=(batch_size, latent_dim)))
            X_real, y_real = np.array(real_batch), np.ones((len(real_batch), 1))
            X_fake, y_fake = fake_images, np.zeros((batch_size, 1))
            d_loss_real = discriminator.train_on_batch(X_real, y_real)
            d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            
            noise = tf.random.normal(shape=(batch_size, latent_dim))
            y_gan = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, y_gan)

        print(f"Epoch {epoch + 1}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")


train_gan(generator, discriminator, gan, real_images, latent_dim, n_epochs=100, batch_size=128)


def generate_images(generator, latent_dim, n_samples):
    noise = tf.random.normal(shape=(n_samples, latent_dim))
    generated_images = generator.predict(noise)
    return generated_images

n_samples = 200
synthetic_images = generate_images(generator, latent_dim, n_samples)


for i, img in enumerate(synthetic_images):
    nii_img = nib.Nifti1Image(img.squeeze(), np.eye(4)) 
    nib.save(nii_img, f"synthetic_image_{i}.nii.gz")
