# Import basic libraries and keras
import os
import keras
import numpy as np
from keras import applications
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Parameters of the model
img_width, img_height = 32, 32
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 1000
nb_validation_samples = 400
epochs = 50
batch_size = 20

# Generator used to load the data
datagen = ImageDataGenerator(rescale=1. / 255)

# Load the VGG16 network from disk
try:
    model = load_model("../ImageRecognition/vgg16_pretrained_imagenet.h5")
except:
    model = applications.VGG16(include_top=False, weights='imagenet')
    model.save("../ImageRecognition/vgg16_pretrained_imagenet.h5")

# Extract bottleneck features
if not os.path.exists('bottleneck_features_train.npy'):
    generator = datagen.flow_from_directory(train_data_dir, shuffle=False,
        target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')
    bottleneck_features_train = model.predict_generator(generator,
        nb_train_samples // batch_size, verbose = 1)
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

if not os.path.exists('bottleneck_features_validation.npy'):
    generator = datagen.flow_from_directory(validation_data_dir, shuffle=False,
        target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')
    bottleneck_features_validation = model.predict_generator(generator,
        nb_validation_samples // batch_size, verbose = 1)
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

# Load bottleneck features
train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
trainquarter = nb_train_samples // 4
train_labels = np.array([0] * trainquarter + [1] * trainquarter + [2] * trainquarter + [3] * trainquarter)
train_labels = keras.utils.to_categorical(train_labels, 4)

validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
validquarter = nb_validation_samples // 4
validation_labels = np.array([0] * validquarter + [1] * validquarter + [2] * validquarter + [3] * validquarter)
validation_labels = keras.utils.to_categorical(validation_labels, 4)

# Use the pretrained features network and add a connected network on top
if not os.path.exists('bottleneck_fc_model.h5'):
    # Create a neural network with 2 dense layers
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,
              verbose=1, validation_data=(validation_data, validation_labels))

    # Save the model
    model.save('bottleneck_fc_model.h5')
else:
    # Load the model from disk
    model = load_model('bottleneck_fc_model.h5')

score = model.evaluate(validation_data, validation_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

