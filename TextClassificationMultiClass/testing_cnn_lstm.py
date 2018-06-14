# Import basic libraries and keras
import json
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the model, either 'cnn_model.h5' or 'lstm_model.h5'
model = load_model('cnn_model.h5')

# Load the dictionary and the model
with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

classes = ['business', 'entertainment', 'politics', 'sport', 'tech']

tokenizer = Tokenizer(num_words=3000)
tokenizer.word_index = dictionary
while 1:
    print("\n\nRelevant types: " + ", ".join(classes))
    text = input('Input a sentence to evaluate its type, or press enter to quit: ')
    if len(text) == 0:
        break
    # Make the prediction
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=300)
    pred = model.predict(padded_sequences, verbose=0)[0]
    category = np.argmax(pred)
    print("The category is %s (confidence: %.2f%%)" % (classes[category], 100 * max(pred)))
