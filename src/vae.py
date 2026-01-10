import tensorflow as tf
import numpy as np

class VAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim = 16, mode = "VAE", use_beta = False):
        super(VAE, self).__init__()
        self.use_beta = use_beta
        self.latent_dim = latent_dim
        if mode.upper() == "VAE":
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(latent_dim * 2)
                ]
            )

            self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(input_dim)
                ]
            )

        if mode.upper() == "CVAE":
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=input_dim),
                    tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(latent_dim + latent_dim),
                ]
            )

            self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(32 * 323 * 16, activation='relu'),  # 16x323x32 from encoder
                    tf.keras.layers.Reshape((32, 323, 16)),  # H, W, channels
                    tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),
                    tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu'),
                    tf.keras.layers.Conv2DTranspose(1, 3, strides=1, padding='same'),  # final reconstruction
                ]
            )

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def __log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)

    def __compute_loss(self, x, beta = 4):
        mean, logvar = self.encode(x)
        logvar = tf.clip_by_value(logvar, -10.0, 10.0)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)

        # Gaussian reconstruction loss
        recon_loss = tf.reduce_mean(tf.square(x - x_recon), axis=1)
        if self.use_beta:
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1))
            loss = recon_loss + beta * kl_loss
        else:
            logpz = self.__log_normal_pdf(z, 0., 0.)
            logqz_x = self.__log_normal_pdf(z, mean, logvar)
            loss = tf.reduce_mean(recon_loss + logqz_x - logpz)

        return loss


    @tf.function
    def __train_step(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.__compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    
    def fit(self, train_dataset, epochs = 10, optimizer = tf.keras.optimizers.Adam(1e-4)):
        for epoch in range(1, epochs + 1):
            loss = tf.keras.metrics.Mean()
            for train_x in train_dataset:
                # Train model
                self.__train_step(train_x, optimizer)
                # Calculate loss
                loss(self.__compute_loss(train_x))
            elbo = -loss.result()
            print(f'Epoch: {epoch}, Training ELBO: {elbo}')
