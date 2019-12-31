import numpy as np
import copy
import os
import pickle
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential
from tetris3d import *
import time
import tensorflow as tf

'''
    A QLearner class based on Jeremy Watt's notes: 
    https://github.com/jermwatt/control-notes/blob/master/posts/deep_q_learning/
    deep_Q_learning/keras_tensorflow/qfitten_deepQlearning_demo_keras.ipynb
    
    Many things have, again, been modified to suit this particular problem. Notably, the generality
    has been stripped away in the functions that construct the model to accomodate the architecture required
    for my simulator.
'''


class TetrisQLearn:

    def __init__(self, games_state, savename, dirname='logs', **kwargs):

        self.simulator = games_state

        # Q learn basic params
        self.explore_val = 1  # probability to explore v exploit
        self.explore_decay = 0.99  # explore chance is reduced as Q resolves
        self.gamma = 1  # short-term/long-term trade-off param
        self.num_episodes = 500  # number of episodes of simulation to perform
        self.save_weight_freq = 10  # controls how often (in number of episodes) the weights of Q are saved
        self.memory = []  # memory container

        # fitted Q-Learning params
        self.episode_update = 1  # after how many episodes should we update Q?
        self.memory_length = 10  # length of memory replay (in episodes)

        def proc(state):
            return state.stripped()

        self.processor = proc

        # let user define each of the params above
        if "gamma" in kwargs:
            self.gamma = kwargs['gamma']
        if 'explore_val' in kwargs:
            self.explore_val = kwargs['explore_val']
        if 'explore_decay' in kwargs:
            self.explore_decay = kwargs['explore_decay']
        if 'num_episodes' in kwargs:
            self.num_episodes = kwargs['num_episodes']
        if 'episode_update' in kwargs:
            self.episode_update = kwargs['episode_update']
        if 'exit_level' in kwargs:
            self.exit_level = kwargs['exit_level']
        if 'exit_window' in kwargs:
            self.exit_window = kwargs['exit_window']
        if 'save_weight_freq' in kwargs:
            self.save_weight_freq = kwargs['save_weight_freq']
        if 'memory_length' in kwargs:
            self.memory_length = kwargs['memory_length']
        if 'episode_update' in kwargs:
            self.episode_update = kwargs['episode_update']
        if 'processor' in kwargs:
            self.processor = kwargs['processor']

        # get simulation-specific variables from simulator
        self.num_actions = self.simulator.output_dimension
        self.state_dim = self.simulator.input_dimension
        self.training_reward = []

        # create text file for training log
        self.logname = os.path.join(dirname, 'training_logs', savename + '.txt')
        self.reward_logname = os.path.join(dirname, 'reward_logs', savename + '.txt')
        self.weight_name = os.path.join(dirname, 'saved_model_weights', savename + '.pkl')
        self.model_name = os.path.join(dirname, 'models', savename + '.json')

        self.init_log(self.logname)
        self.init_log(self.reward_logname)
        self.init_log(self.weight_name)
        self.init_log(self.model_name)

    # Logging stuff
    def init_log(self, logname):
        # delete log if old version exists
        if os.path.exists(logname):
            os.remove(logname)

    def update_log(self, logname, update):
        if type(update) == str:
            logfile = open(logname, "a")
            logfile.write(update)
            logfile.close()
        else:
            weights = []
            if os.path.exists(logname):
                with open(logname, 'rb') as rfp:
                    weights = pickle.load(rfp)
            weights.append(update)

            with open(logname, 'wb') as wfp:
                pickle.dump(weights, wfp)

    def init_Q_from_model(self, model, processor=None, **kwargs):
        if processor:
            self.processor = processor

        loss = 'mse'
        lr = 10 ** (-2)
        if 'alpha' in kwargs:
            lr = kwargs['alpha']
        if 'loss' in kwargs:
            loss = kwargs['alpha']
        optimizer = RMSprop(lr=lr)
        if 'optimizer' in kwargs:
            optimizer = kwargs['optimizer'](lr=lr)

        self.model = model
        self.model.compile(optimizer=optimizer, loss=loss)
        self.Q = self.model.predict

        # since its easy to save models with keras / tensorflow, save to file
        model_json = self.model.to_json()
        with open(self.model_name, "w") as json_file:
            json_file.write(model_json)

    # Q Learning Stuff
    def initialize_Q(self, **kwargs):
        # by default a 3 layer (with a flattening layer) fully connected network
        layer_sizes = [10, 10]
        activation = 'relu'
        if 'layer_sizes' in kwargs:
            layer_sizes = kwargs['layer_sizes']
        if 'activation' in kwargs:
            activation = kwargs['activation']
        loss = 'mse'
        lr = 10 ** (-2)
        if 'alpha' in kwargs:
            lr = kwargs['alpha']
        if 'loss' in kwargs:
            loss = kwargs['alpha']
        optimizer = RMSprop(lr=lr)
        if 'optimizer' in kwargs:
            optimizer = kwargs['optimizer'](lr=lr)

        # Input/Output size fot the network
        input_dim = self.state_dim
        output_dim = self.num_actions

        # build the model
        self.model = Sequential()

        # add input layer
        self.model.add(Dense(layer_sizes[0], input_dim=input_dim, activation=activation))

        # add hidden layers
        for U in layer_sizes[1:]:
            self.model.add(Dense(U, activation=activation))

        # add output layer
        self.model.add(Dense(output_dim, activation='linear'))

        # chose optimizer and its associated parameters
        self.model.compile(loss=tf.losses.huber_loss, optimizer=optimizer)

        # initialize Q
        self.Q = self.model.predict

        # since its easy to save models with keras / tensorflow, save to file
        model_json = self.model.to_json()
        with open(self.model_name, "w") as json_file:
            json_file.write(model_json)

    def memory_replay(self):
        # these are "phantom" regressors generated from previous Q function
        q_vals = []
        states = []
        for i in range(len(self.memory)):
            episode_data = self.memory[i]

            for j in range(len(episode_data)):
                sample = episode_data[j]

                # strip sample for parts
                state = sample[0]
                next_state = sample[1]
                action = sample[2]
                reward = sample[3]
                done = sample[4]

                done, reward = self.check_done(done, reward)

                q = reward
                if not done:
                    qs = self.Q(next_state)
                    q += self.gamma * np.max(qs)

                # clamp all other models to their current values for this input/output pair
                q_update = self.Q(state).flatten()
                q_update[action] = q
                q_vals.append(q_update)
                states.append(state.T)

        # convert lists to numpy arrays for regressor
        s_in = np.array(states).T
        q_vals = np.array(q_vals).T
        s_in = s_in[0, :, :]

        # take descent step
        num_pts = s_in.shape[1]
        self.model.fit(x=s_in.T, y=q_vals.T, batch_size=num_pts, epochs=1, verbose=0)

        # update Q based on regressor updates
        self.Q = self.model.predict

    def check_done(self, done, reward):
        if done:
            reward = self.simulator.game_over_penalty
        return done, reward

    def update_memory(self, episode_data):
        # add most recent trial data to memory
        self.memory.append(episode_data)

        # clip memory if it gets too long
        num_episodes = len(self.memory)
        if num_episodes >= self.memory_length:
            num_delete = num_episodes - self.memory_length
            self.memory[:num_delete] = []

    # choose next action
    def choose_action(self, state):
        # pick action at random
        p = np.random.rand(1)
        action = np.random.randint(self.num_actions)

        # pick action based on exploiting
        qs = self.Q(state)

        if p > self.explore_val:
            action = np.argmax(qs)
        return action

    # assume the state has been preprocessed
    def state_normalizer(self, states):
        states = np.array(states)[np.newaxis, :]
        return states

    def run(self):

        print("num_episodes: %s" % self.num_episodes)

        # start main Q-learning loop
        for n in range(self.num_episodes):
            # pick this episode's starting position - randomly initialize from f_system
            state = self.simulator.reset()
            state = self.state_normalizer(self.processor(state))
            total_episode_reward = 0
            done = False

            # get our exploit parameter for this episode
            if self.explore_val > 0.01:
                self.explore_val *= self.explore_decay

            # run episode
            step = 0
            episode_data = []

            ep_start_time = time.time()
            while done is False:

                # choose next action

                action = self.choose_action(state)

                # transition to next state, get associated reward
                next_state, reward, done = self.simulator(self.simulator.action_space[action])
                next_state = self.state_normalizer(self.processor(next_state))

                # store data for transition after episode ends
                episode_data.append([state, next_state, action, reward, done])

                # update total reward from this episode
                total_episode_reward += reward
                state = copy.deepcopy(next_state)
                step += 1

            # update memory with this episode's data
            self.update_memory(episode_data)

            curtime = time.time()
            sim_time = curtime - self.simulator.start_time

            # update Q function
            if np.mod(n, self.episode_update) == 0:
                self.memory_replay()

            fit_time = time.time() - curtime

            # update episode reward greater than exit_level, add to counter
            exit_ave = total_episode_reward
            if n >= self.exit_window:
                exit_ave = np.sum(np.array(self.training_reward[-self.exit_window:])) / self.exit_window

            # print out updates
            update = 'episode ' + str(n + 1) + ' of ' + str(
                self.num_episodes) + ' complete, ' + ' explore val = ' + str(
                np.round(self.explore_val, 3)) + ', episode reward = ' + str(
                np.round(total_episode_reward, 1)) + ', ave reward = ' + str(
                np.round(exit_ave, 1)) + ', sim time = ' + str(
                np.round(sim_time, 3)) + ', fit time = ' + str(
                np.round(fit_time, 3)) + ', episode_time = ' + str(np.round(time.time() - ep_start_time, 3))

            print(update)
            self.update_log(self.logname, update + '\n')

            update = str(total_episode_reward) + '\n'
            self.update_log(self.reward_logname, update)

            # store this episode's computation time and training reward history
            self.training_reward.append(total_episode_reward)

            # save latest weights from this episode
            if np.mod(n, self.save_weight_freq) == 0:
                update = self.model.get_weights()
                self.update_log(self.weight_name, update)

        # save weights
        update = 'q-learning algorithm complete'
        self.update_log(self.logname, update + '\n')
        print(update)
