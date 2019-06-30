import numpy as np
from keras import layers, models, optimizers

from keras.utils import to_categorical
from keras import callbacks

from keras import backend as K
# change the format of the color channel (28, 28, 1)
K.set_image_data_format('channels_last')

epochs = 32
batch_size = 32
lr =1e-4

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train.astype('float32'))
y_test = to_categorical(y_test.astype('float32'))

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = models.Sequential([
    layers.Flatten(input_shape=(28,28,1)),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=optimizers.Adam(lr=lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

tb = callbacks.TensorBoard(log_dir='tensorboard-logs', batch_size=32, histogram_freq=10)

#Training without data augmentation
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          validation_data=[x_test, y_test], callbacks=[tb])

model.save_weights('trained_model.h5')
print('Trained model saved to trained_model.h5')


