# Import basic libraries and keras
import json
import numpy as np
from keras.models import load_model
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer

# Load the dictionary and the model
with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)
model = load_model('model.h5')

classes = ['business', 'entertainment', 'politics', 'sport', 'tech']

tokenizer = Tokenizer(num_words=3000)
while 1:
    print("\n\nRelevant types: " + ", ".join(classes))
    text = input('Input a sentence to evaluate its type, or press enter to quit: ')
    if len(text) == 0:
        break
    # Make the prediction
    words = kpt.text_to_word_sequence(text)
    wordIndices = [dictionary[word] for word in words if word in dictionary]
    testdata = tokenizer.sequences_to_matrix([wordIndices], mode='binary')
    pred = model.predict(testdata)[0]
    category = np.argmax(pred)
    print("The category is %s (confidence: %.2f%%)" % (classes[category], 100 * max(pred)))
