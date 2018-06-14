import os
import numpy as np
import matplotlib.pyplot as plt

from keras.layers.core import Dense
from keras.models import Sequential, load_model
from keras.layers.advanced_activations import PReLU

from qlearning import qtrain

plt.ion()
plt.show()


maze =  np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  0.,  1.,  0.],
    [ 0.,  0.,  0.,  1.,  1.,  1.,  0.],
    [ 1.,  1.,  1.,  1.,  0.,  0.,  1.],
    [ 1.,  0.,  0.,  0.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.]
])

# Check if there is a pre-trained model
if not os.path.exists('model.h5'):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(4)) # num of actions
    model.compile(optimizer='adam', loss='mse')

    model = qtrain(model, maze, n_epoch=1000, max_memory=8*maze.size, data_size=32, visualize = False)
    # Save the model
    model.save('model.h5')
else:
    # Load the model from disk
    model = load_model('model.h5')

from qmaze import Qmaze, play_game
play_game(model, Qmaze(maze), (0, 0), True)


