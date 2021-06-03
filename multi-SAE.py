import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Dense, Input
from keras.layers import Convolution2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model, load_model
from keras import backend as K
from keras.datasets import mnist
import os


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

(x_train, y_train_), (x_test, y_test_) = mnist.load_data('mnist.npz')
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

input_shape = (image_size, image_size, 1)
batch_size = 256
kernel_size = 3
filters = 16
latent_dim = 2
epochs = 30

x_in = Input(shape=input_shape)
x = x_in
for i in range(2):
    filters *=2
    x = Convolution2D(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        strides=2,
        padding='same',
    )(x)


shape = K.int_shape(x)

x = Flatten()(x)  # 把多维的数据变为一维
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + epsilon * K.exp(z_log_var/2)

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

##### 定义模型解码部分（生成器） #####

latent_inputs = Input(shape=(latent_dim,))
x = Dense(shape[1]*shape[2]*shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(2):
    x = Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        strides=2,
        padding='same'
    )(x)
    filters //=2

outputs = Conv2DTranspose(
    filters=1,
    kernel_size=kernel_size,
    activation='sigmoid',
    padding='same'
)(x)

encoder = Model(x_in, z_mean)
decoder = Model(latent_inputs, outputs)
x_out = decoder(z)
vae = Model(x_in, x_out)


xent_loss = K.sum(K.binary_crossentropy(x_in, x_out), axis=[1, 2, 3])
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)

vae.compile(optimizer='rmsprop')
vae.summary()


vae.fit(
    x_train,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, None)
)
vae.save('./model/model_vae_cnn')
encoder.save('./model/model_vae_cnn_encoder')
decoder.save('./model/model_vae_cnn_decoder')


encoder_model = load_model('./model/model_vae_cnn_encoder', custom_objects={'sampling':sampling})
x_test_encoded = encoder_model.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6,6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test_)
plt.colorbar()
plt.show()


decoder_model = load_model('./model/model_vae_generator', custom_objects={'sampling':sampling})
n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder_model.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()













