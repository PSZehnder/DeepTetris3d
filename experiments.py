from learner import *
import os
from tetris3d import *
import matplotlib.pyplot as plt
import numpy as np
from render import plot_reward_history
import pickle

dir = os.getcwd()
save_name = '219-oleg-continued'

# resume from 0.012

# tetris settings
clear_reward = 100 #100
pieces_reward = 0.003 #0.05 #0.05 # 0.05 -> 1.1 to 2
packing_reward = 0.035 #0.004 #0.004 -> 1.5 to 2.4
variance_penalty = 0.00075 #0.0005 #0.0005
game_over_penalty = 7.5
game_len_reward = 0.0009
height_penalty = 0.00065
empty_column_penalty = 0.0045
overhang_penalty = 0.00004

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

board_shape = [7, 7, 15]

tetris_instance = GameState(board_shape=board_shape, rewards=rewards)

# Q learn settings
num_episodes = 25000
explore_decay = 0.997
explore_val = 0.90
exit_level = 200
exit_window = 4
save_weight_freq = 100
target_refresh = 256
minibatch_size = 1024
bsize = 64

# initialize memory
episode_update = 4
memory_length = 256

# load into instance of learner
learner = TetrisQLearn(tetris_instance, save_name, dir,
                       gamma=0.9,
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
                       minibatch_size=minibatch_size)

# initialize Q function
N = 128
alpha = 10**(-5)
learner.initialize_Q(alpha=alpha, model_path='saved_model_weights/219-oleg/219-oleg200.pth',
                     in_channels=N, neurons=256, fc_channels=N, dense_channels=N,
                     concat_channels=N, pool=7)

learner.run()
reward_logname = os.path.join('reward_logs', save_name + '.txt')
plot_reward_history(reward_logname, window_length=exit_window)

