from learner import *
import os
from tetris3d import *
import matplotlib.pyplot as plt
import numpy as np
from render import plot_reward_history
from keras.models import model_from_json
import tensorflow as tf

dir = os.getcwd()
save_name = 'oleg_last_try_2'


# bespoke loss function that's supposed to help with network convergence since it combines
# MSE and MAE, generally speaking
def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)


# tetris settings
clear_reward = 10
clear_exponent = 2
pieces_reward = 0.25
packing_penalty = 0.07
holes_penalty = packing_penalty / 2
game_over_penalty = -500

board_shape = [5, 5, 20]

tetris_instance = GameState(board_shape=board_shape, rewards=[clear_reward, clear_exponent, pieces_reward,
                                                              packing_penalty, holes_penalty, game_over_penalty])

# Q learn settings
num_episodes = 100000
explore_decay = 1
explore_val = 0.01
exit_level = 200
exit_window = 100
save_weight_freq = 1000

# initialize memory
episode_update = 10
memory_length = 15

# load into instance of learner
learner = TetrisQLearn(tetris_instance, save_name, dir,
                       num_episodes=num_episodes,
                       explore_decay=explore_decay,
                       explore_val=explore_val,
                       memory_length=memory_length,
                       episode_update=episode_update,
                       exit_level=exit_level,
                       exit_window=exit_window,
                       save_weight_freq=save_weight_freq)

# initialize Q function
layer_sizes = [500, 1000]
alpha = 10**(-2)
activation = 'relu'
learner.initialize_Q(layer_sizes=layer_sizes, alpha=alpha, activation=activation, loss=tf.losses.huber_loss)

# Here's an example of how to load up a model and continue training
# with open('models/oleg_last_try.json', 'r') as mod:
#     model = model_from_json(mod.read())
#
# weights = pickle.load(open("saved_model_weights/oleg_last_try.pkl", 'rb'))
#
# model.set_weights(weights[len(weights) -1])
#
# learner.init_Q_from_model(model)

# do the damn thing
learner.run()
reward_logname = os.path.join('reward_logs', save_name + '.txt')
plot_reward_history(reward_logname, window_length=exit_window)

