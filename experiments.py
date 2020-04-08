from learner import *
import os
from tetris3d import *
import matplotlib.pyplot as plt
import numpy as np
from render import plot_reward_history
import pickle

dir = os.getcwd()
save_name = '220-oleg-packing_eff_only'

# resume from 0.012

### THESE ARE ALL POSITIVE SIGNALS EXCEPT GAME OVER
# tetris settings
clear_reward = 1000
pieces_reward = 0
packing_reward = 15
variance_penalty = 0 #0.085
game_over_penalty = 50
game_len_reward = 0
height_penalty = 0 #0.02
empty_column_penalty = 0 #1.1
overhang_penalty = 0

rewards = {
            'clear_reward'      : clear_reward,
            'pieces_reward'     : pieces_reward,
            'packing_reward'    : packing_reward,
            'variance_penalty'  : variance_penalty,
            'game_over_penalty' : game_over_penalty,
            'game_len_reward'   : game_len_reward,
            'empty_column_penalty' : empty_column_penalty,
            'height_penalty' : height_penalty,
            'overhang_penalty': overhang_penalty
           }

with open(os.path.join('reward_logs', save_name + '-reward_dict.pkl'), 'wb') as pick_file:
    pickle.dump(rewards, pick_file)

board_shape = [10, 10, 20]

tetris_instance = GameState(board_shape=board_shape, rewards=rewards)

# Q learn settings
num_episodes = 25000
explore_decay = 0.995
explore_val = 1
exit_level = 200
exit_window = 4
save_weight_freq = 200
target_refresh = 256
minibatch_size = 128
bsize = 64
use_target = True

# initialize memory
episode_update = 2
memory_length = 8

# load into instance of learner
learner = TetrisQLearn(tetris_instance, save_name, dir,
                       gamma=0.95,
                       num_episodes=num_episodes,
                       explore_decay=explore_decay,
                       explore_val=explore_val,
                       batch_size=bsize,
                       memory_length=memory_length,
                       episode_update=episode_update,
                       exit_level=exit_level,
                       exit_window=exit_window,
                       save_weight_freq=save_weight_freq,
                       schedule=True,
                       refresh_target=target_refresh,
                       minibatch_size=minibatch_size,
                       render_path='images',
                       use_target=False)

# initialize Q function
modelpath = None
N = 128
alpha = 10**(-5)
learner.initialize_Q(alpha=alpha, model_path= modelpath,
                     in_channels=N, neurons=256, fc_channels=N, dense_channels=N,
                     concat_channels=N, pool=7)

learner.run()
reward_logname = os.path.join('reward_logs', save_name + '.txt')
plot_reward_history(reward_logname, window_length=exit_window)

