# Import basic libraries and keras
import os
import json
import keras
import numpy as np
import keras.preprocessing.text as kpt
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from dataload import load_data

# Load input data and labels (0 to 4)
train_x, train_y = load_data()

# Use the 3000 most popular words found in our dataset
max_words = 3000

# Tokenize the data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_x)
dictionary = tokenizer.word_index
# Save tokenizer dictionary to file
if not os.path.exists('dictionary.json'):
    with open('dictionary.json', 'w') as outfile:
        json.dump(tokenizer.word_index, outfile)

# For each text, change each token to its ID in the Tokenizer's word_index
allWordIndices = []
for text in train_x:
    words = kpt.text_to_word_sequence(text)
    wordIndices = [dictionary[word] for word in words]
    allWordIndices.append(wordIndices)

# Create matrix with indexed texts and categorical target
train_x = tokenizer.sequences_to_matrix(np.asarray(allWordIndices), mode='binary')
train_y = keras.utils.to_categorical(train_y, 5)

# Check if there is a pre-trained model
if not os.path.exists('model.h5'):
    # Create a neural network with 3 dense layers
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    model.fit(train_x, train_y, batch_size=32, epochs=10, verbose=1, validation_split=0.1, shuffle=True)

    # Save the model
    model.save('model.h5')
else:
    # Load the model from disk
    model = load_model('model.h5')

