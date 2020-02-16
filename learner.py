import numpy as np
import copy
import os
import pickle
from tetris3d import *
import time
import torch.optim as optim
import torch.nn as nn
import torch
from model import DenseNet
from torch.utils.data import Dataset, DataLoader
from termcolor import colored
import csv
from shapes3d import embed
import random

def add_reward_dicts(dict1, dict2):
    new_dict = {}
    for k, v in dict1.items():
        new_dict[k] = v + dict2[k]
    return new_dict

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
        self.explore_decay = 0.999  # explore chance is reduced as Q resolves
        self.gamma = 1  # short-term/long-term trade-off param
        self.num_episodes = 500  # number of episodes of simulation to perform
        self.save_weight_freq = 10  # controls how often (in number of episodes) the weights of Q are saved
        self.memory = []
        self._process_mask = []# memory container

        # fitted Q-Learning params
        self.episode_update = 1  # after how many episodes should we update Q?
        self.batch_size = 10  # length of memory replay (in episodes)

        self.schedule = False
        self.refresh_target = 1

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
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        if 'episode_update' in kwargs:
            self.episode_update = kwargs['episode_update']
        if 'schedule' in kwargs:
            self.schedule = kwargs['schedule']
        if 'memory_length' in kwargs:
            self.memory_length = kwargs['memory_length']
        if 'refresh_target' in kwargs:
            self.refresh_target = kwargs['refresh_target']
        if 'minibatch_size' in kwargs:
            self.minibatch_size = kwargs['minibatch_size']

        # get simulation-specific variables from simulator
        self.num_actions = self.simulator.output_dimension
        self.training_reward = []

        # create text file for training log
        self.logname = os.path.join(dirname, 'training_logs', savename + '.txt')
        self.reward_logname = os.path.join(dirname, 'reward_logs', savename + '.txt')
        if not os.path.exists(os.path.join(dirname, 'saved_model_weights', savename)):
            os.mkdir(os.path.join(dirname, 'saved_model_weights', savename))
        self.weights_folder = os.path.join(dirname, 'saved_model_weights', savename)

        self.reward_table = os.path.join(dirname, 'reward_logs_extended', savename + '.csv')
        self.savename = savename
        self.weights_idx = 0

        self.init_log(self.logname)
        self.init_log(self.reward_logname)
        self.init_log(self.reward_table)

        self.write_header = True

    # Logging stuff
    def init_log(self, logname):
        # delete log if old version exists
        if os.path.exists(logname):
            os.remove(logname)

    def update_log(self, logname, update, epoch=None):
        if type(update) == str:
            logfile = open(logname, "a")
            logfile.write(update)
            logfile.close()
        else:
            mod = self.model.state_dict()
            torch.save(mod, os.path.join(self.weights_folder, self.savename + str(epoch) + '.pth'))
            self.weights_idx += 1

    def log_reward(self, reward_dict):

        keys = reward_dict.keys()

        if self.write_header:
            with open(self.reward_table, 'a') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
            self.write_header = False


        with open(self.reward_table, 'a') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writerow(reward_dict)

    # Q Learning Stuff
    def initialize_Q(self, alpha=None, **kwargs):
        lr = 10 ** (-2)
        if alpha:
            lr = alpha

        # Input/Output size fot the network
        output_dim = self.num_actions

        self.model = DenseNet(output_dim, **kwargs)
        self.Q = self.model
        self.target_network = self.Q.copy()

        self.use_cuda = False
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.use_cuda = True
            self.model.to(torch.device('cuda:0'))
            self.device = torch.device('cuda:0')

        self.target_network.to(self.device)
        self.model.to(self.device)

        self.loss = nn.SmoothL1Loss()
        self.max_lr = lr
        self.optimizer = optim.RMSprop(self.model.parameters(), lr)
        if self.schedule:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10,
                                                                  verbose=True)
        for p in self.model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    def _embed_piece(self, piece, embedding_size=(4, 4, 4)):
        out = np.zeros((piece.shape[0], piece.shape[1], *embedding_size))
        for coord in product(*[range(x) for x in piece.shape]):
            out[coord] = piece[coord]
        return torch.from_numpy(out).to(self.device)

    def memory_replay(self):
        # process transitions using target network
        q_vals = None
        states = []
        pieces = []
        locats = []
        episode_loss = 0

        for i in range(len(self.memory)):
            if not self._process_mask[i]:
                self._process_mask[i] = True
                episode_data = self.memory[i]

                for j in range(len(episode_data)):
                    sample = episode_data[j]

                    # strip sample for parts
                    state, piece, locat = sample[0]
                    next_state, next_piece, next_locat = sample[1]
                    action = sample[2]
                    reward = sample[3]
                    done = sample[4]

                    q = reward
                    if not done:
                        next_state = torch.tensor(next_state).to(self.device)
                        next_piece = torch.tensor(next_piece).to(self.device)
                        next_locat = torch.tensor(next_locat).to(self.device)

                        qs = self.target_network(next_state, next_piece, next_locat)
                        q += self.gamma * torch.max(qs)

                    # clamp all other models to their current values for this input/output pair
                    state = torch.tensor(state).to(self.device)
                    piece = torch.tensor(piece).to(self.device)
                    locat = torch.tensor(locat).to(self.device)

                    q_update = self.target_network(state, piece, locat).squeeze(0)
                    q_update[action] = q
                    if q_vals is None:
                        q_vals = [q_update.detach().cpu().numpy()]
                    q_vals = q_vals + [q_update.detach().cpu().numpy()]
                    states.append(state.detach())
                    pieces.append(piece.float().squeeze(0).detach())
                    locats.append(locat.float().detach())

        self.Q.train()

        # convert to tensor of right size
        s_in = [s for game in states for s in game]

        # take descent step
        memory = MemoryDset(s_in, pieces, locats, q_vals)

        if self.minibatch_size > 0:
            ids = random.sample(range(len(memory)), min(self.minibatch_size, len(memory) - 1))
            memory = torch.utils.data.Subset(memory, ids)
        dataloader = DataLoader(memory, batch_size=self.batch_size, shuffle=True)

        tick = time.time()
        for s, p, l, q in dataloader:
            self.optimizer.zero_grad()

            out = self.Q(s, p, l)
            loss = self.loss(out, q)
            loss.backward()
            self.optimizer.step()
            episode_loss += loss.item()
            s.detach_()
        print('process time: %.2f' % (time.time() - tick))
        print(len(dataloader.dataset))
        
        if self.schedule:
            self.scheduler.step(episode_loss)
        return episode_loss / len(self.memory) / len(dataloader)

    def update_target(self):
        self.target_network = self.Q.copy()
        self.target_network.to(self.device)
        self._process_mask = [False] * len(self._process_mask)  # reset the processed_mask

    def update_memory(self, episode_data):
        # add most recent trial data to memory
        self.memory.append(episode_data)
        self._process_mask.append(False)

        # clip memory if it gets too long
        num_episodes = len(self.memory)
        if num_episodes >= self.memory_length:
            num_delete = num_episodes - self.memory_length
            self.memory[:num_delete] = []
            self._process_mask[:num_delete] = []

    def make_torch(self, array):
        tens = torch.from_numpy(array.copy())
        tens = tens.float()
        tens = tens.unsqueeze(0)
        tens = tens.unsqueeze(0)
        return  tens.to(self.device)

    # choose next action
    def choose_action(self, state, piece, location):
        invalid_actions = []
        for i in range(len(self.simulator.action_space)): #todo paralellize
            invalid_actions.append(not self.simulator.is_valid_action(self.simulator.action_space[i]))
        invalid_actions = np.array(invalid_actions)
        # pick action at random
        p = np.random.rand(1)

        action = np.random.randint(len(self.simulator.action_space))
        while invalid_actions[action]:
            action = np.random.randint(len(self.simulator.action_space))

        # pick action based on exploiting
        qs = self.Q(state, piece, location)
        qs = self.renormalize_vec(qs, invalid_actions)

        if p > self.explore_val:
            action = torch.argmax(qs)
        return action

    def renormalize_vec(self, tensor, idx):
        tensor = tensor.squeeze(0)
        tensor[idx] = 0
        sum = torch.sum(tensor)
        return tensor / sum

    def run(self):

        print("num_episodes: %s" % self.num_episodes)

        # start main Q-learning loop
        for n in range(self.num_episodes):
            # pick this episode's starting position
            state = self.simulator.reset()
            total_episode_reward = 0
            done = False

            # get our exploit parameter for this episode
            if self.explore_val > 0.01 and (n % self.episode_update) == 0:
                old_explore = self.explore_val
                self.explore_val *= self.explore_decay
                if old_explore - self.explore_val > 0.25:
                    for param in self.optimizer.param_groups:
                        print('resetting to max learning rate: %s' % self.max_lr)
                        param['lr'] = self.max_lr

            # run episode
            step = 0
            episode_data = []
            ep_start_time = time.time()
            ep_rew_dict = None

            while done is False:

                # choose next action
                board = self.make_torch(state.board)
                piece = self.make_torch(state.current.matrix)
                loc =  torch.tensor(state.current.location).unsqueeze(0).to(self.device)
                action = self.choose_action(board, piece, loc)

                # transition to next state, get associated reward
                next_state, reward_dict, done = self.simulator(self.simulator.action_space[action])
                if ep_rew_dict is None:
                    ep_rew_dict = reward_dict
                else:
                    ep_rew_dict = add_reward_dicts(ep_rew_dict, reward_dict)
                next_board = self.make_torch(next_state.board)
                next_piece = self.make_torch(next_state.current.matrix)
                next_locat = torch.tensor(next_state.current.location).unsqueeze(0)

                reward = reward_dict['total']

                # move board back to cpu to clear up vram
                board = board.cpu().numpy()
                next_board = next_board.cpu().numpy()
                piece = piece.cpu().numpy()
                location = loc.cpu().numpy()
                next_piece = next_piece.cpu().numpy()
                next_locat = next_locat.cpu().numpy()

                # store data for transition after episode ends
                episode_data.append([(board, piece, location), (next_board, next_piece, next_locat), action, reward, done])

                # update total reward from this episode
                total_episode_reward += reward
                state = copy.deepcopy(next_state)
                step += 1

            # update memory with this episode's data
            self.update_memory(episode_data)

            LOSS_SCALING  = 100

            # update the target network
            if np.mod(n, self.refresh_target) == 0:
                self.update_target()

            # train model
            episode_loss = 0
            if np.mod(n, self.episode_update) == 0:
                episode_loss = self.memory_replay() * LOSS_SCALING

            # update episode reward greater than exit_level, add to counter
            exit_ave = total_episode_reward
            if n >= self.exit_window:
                exit_ave = np.sum(np.array(self.training_reward[-self.exit_window:])) / self.exit_window

            # print out updates
            # I abuse the fuck out of this variable. Watch how many different values it assumes and how
            # important the order of operations is. I do this because I hate myself.
            update = 'episode ' + str(n + 1) + ' of ' + str(
                self.num_episodes) + ' complete, ' + 'loss x%s = ' % LOSS_SCALING +str(
                np.round(episode_loss, 3)) + ' explore val = ' + str(
                np.round(self.explore_val, 3)) + ', episode reward = ' + str(
                np.round(total_episode_reward, 1)) + ', ave reward = ' + str(
                np.round(exit_ave, 3)) + ', episode_time = ' + str(np.round(time.time() - ep_start_time, 3))

            self.update_log(self.logname, update + '\n')

            if np.mod(n, self.episode_update) == 0:
                print(colored(update, 'red'))
            else:
                print(update)

            # save latest weights from this episode
            if np.mod(n, self.save_weight_freq) == 0:
                update = self.model.state_dict()
                self.update_log(self.weights_folder, update, epoch=n)

            update = str(total_episode_reward) + '\n'
            self.update_log(self.reward_logname, update)
            self.log_reward(ep_rew_dict)

            # store this episode's computation time and training reward history
            self.training_reward.append(total_episode_reward)

        update = 'q-learning algorithm complete'
        self.update_log(self.logname, update + '\n')
        print(update)

class MemoryDset(Dataset):

    def __init__(self, *lists, device=torch.device('cuda')):

        for i, lst in enumerate(lists):
            self.__setattr__('t%d' % i, lst)

        self.lists = lists
        self.device = device

    def __len__(self):
        return len(self.t0)

    def __getitem__(self, item):
        out = []
        for lst in self.lists:
            if not isinstance(lst[item], torch.Tensor):
                t = torch.tensor(lst[item])
            else:
                t = lst[item]
            t = t.to(self.device)
            out.append(t)
        return out