import os
from tetris3d import *
import time
import torch.optim as optim
import torch.nn as nn
import torch
from model import DenseNet
from torch.utils.data import Dataset, DataLoader
from termcolor import colored
import csv
import random
import copy
import render
import matplotlib.pyplot as plt

# TODO: 2-21 fix 'use_target'

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
        self._process_mask = []
        self.processed_memory = []# memory container

        # fitted Q-Learning params
        self.episode_update = 1  # after how many episodes should we update Q?
        self.batch_size = 10  # length of memory replay (in episodes)

        self.schedule = False
        self.refresh_target = 1

        self.renderpath = None

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
        if 'render_path' in kwargs:
            self.renderpath = kwargs['render_path']
        if 'use_target' in kwargs:
            self.use_target = kwargs['use_target']

        # get simulation-specific variables from simulator
        self.num_actions = self.simulator.output_dimension
        self.training_reward = []
        self.savename = savename

        # create text file for training log
        self.logname = os.path.join(dirname, 'training_logs', savename + '.txt')
        self.reward_logname = os.path.join(dirname, 'reward_logs', savename + '.txt')
        if not os.path.exists(os.path.join(dirname, 'saved_model_weights', savename)):
            os.mkdir(os.path.join(dirname, 'saved_model_weights', savename))
        if self.renderpath and not os.path.exists(os.path.join(self.renderpath, self.savename)):
            os.makedirs(os.path.join(self.renderpath, self.savename), exist_ok=True)
        self.renderpath = os.path.join(self.renderpath, self.savename)

        self.weights_folder = os.path.join(dirname, 'saved_model_weights', savename)
        self.reward_table = os.path.join(dirname, 'reward_logs_extended', savename + '.csv')
        self.weights_idx = 0

        self.init_log(self.logname)
        self.init_log(self.reward_logname)
        self.init_log(self.reward_table)

        self.write_header = True

    def render_model(self, epoch):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        print('rendering...')
        tick = time.time()
        axis_extents = self.simulator.board_extents
        ax.set_xlim3d(0, axis_extents[0])
        ax.set_ylim3d(0, axis_extents[1])
        ax.set_zlim3d(0, axis_extents[2])
        demo_game = GameState(board_shape=axis_extents, rewards=self.simulator.rewards)
        self.model.eval()
        path = os.path.join(self.renderpath, self.savename + '-' + str(epoch) + '.gif')
        render.render_from_model(self.model, fig, ax, demo_game, path, device=self.device)
        self.model.train()
        print('rendered %s in %.2fs' % (path, time.time() - tick))

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
    def initialize_Q(self, model_path=None, alpha=None, **kwargs):
        lr = 10 ** (-2)
        if alpha:
            lr = alpha

        # Input/Output size fot the network
        output_dim = self.num_actions

        self.model = DenseNet(output_dim, **kwargs)
        if model_path is not None:
            print('loading check point %s' % model_path)
            self.model.load_state_dict(torch.load(model_path))
        self.Q = self.model

        self.target_network = self.Q.copy()

        # self.target_network = self.Q

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
        q_vals = []
        states = []
        pieces = []
        locats = []
        episode_loss = 0

        total_processed = 0

        tick = time.time()
        for i in range(len(self.memory)):
            episode_data = self.memory[i]
            if self.processed_memory[i] is None:
                self.processed_memory[i] = [None] * len(episode_data)

            for j in range(len(episode_data)):
                # process the sample and put it into the processed_memory
                sample = episode_data[j]

                state, piece, locat = sample[0]
                next_state, next_piece, next_locat = sample[1]
                action = sample[2]
                reward = sample[3]
                done = sample[4]

                if self.processed_memory[i][j] is None:
                    q = reward

                    # preprocess q using target network
                    if not done:
                        next_state = torch.tensor(next_state).to(self.device)
                        next_piece = torch.tensor(next_piece).to(self.device)
                        next_locat = torch.tensor(next_locat).to(self.device)

                        qs = self.target_network(next_state, next_piece, next_locat) #should be target
                        q += self.gamma * torch.max(qs)

                    state = torch.tensor(state).to(self.device)
                    piece = torch.tensor(piece).to(self.device)
                    locat = torch.tensor(locat).to(self.device)

                    # q is our experientially validated move score. Anchor it on our prediction vector
                    q_update = self.target_network(state, piece, locat).squeeze(0) # should be target
                    q_update[action] = q
                    processed = q_update.detach().cpu().numpy()
                    q_vals = q_vals + [processed]
                    self.processed_memory[i][j] = processed
                    total_processed += 1

                    ## WE HAVE NOW PREPROCESSED W TARGET NETWORK

                    # clear up the vram
                    state = state.cpu().squeeze(0).numpy()
                    piece = piece.float().squeeze(0).cpu().numpy()
                    locat = locat.float().cpu().numpy()

                else:
                    q_vals = q_vals + [self.processed_memory[i][j]]

                # its goofy but it will work
                if state.ndim > 4:
                    state = state.squeeze(0)
                assert state.ndim == 4

                if piece.ndim > 4:
                    piece = piece.squeeze(0)
                assert piece.ndim == 4

                if locat.ndim > 2:
                    locat = locat.squeeze(0)
                assert locat.ndim == 2

                states.append(state)
                pieces.append(piece)
                locats.append(locat)

        elapsed_time = time.time() - tick
        print('process time: %.2f' % elapsed_time)
        print('total processed: %d (%.2f/s)' % (total_processed, total_processed / elapsed_time))

        # take descent step
        memory = MemoryDset(states, pieces, locats, q_vals)

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
        print('fit time: %.2f' % (time.time() - tick))
        
        if self.schedule:
            self.scheduler.step(episode_loss)
        return episode_loss / len(self.memory) / len(dataloader)

    def update_target(self):
        self.target_network = self.Q.copy()
        self.target_network.to(self.device)
        self.processed_memory = [None] * len(self.processed_memory)

    def update_memory(self, episode_data):
        # add most recent trial data to memory
        self.memory.append(episode_data)
        self.processed_memory.append(None)

        # clip memory if it gets too long
        num_episodes = len(self.memory)
        if num_episodes >= self.memory_length:
            num_delete = num_episodes - self.memory_length
            self.memory[:num_delete] = []
            self.processed_memory[:num_delete] = []

    def make_torch(self, array):
        tens = torch.from_numpy(array.copy())
        tens = tens.float()
        tens = tens.unsqueeze(0)
        tens = tens.unsqueeze(0)
        return  tens.to(self.device)

    # choose next action
    def choose_action(self, state, piece, location):
        # pick action at random
        p = np.random.rand(1)

        action = np.random.randint(len(self.simulator.action_space))

        # pick action based on exploiting
        qs = self.Q(state, piece, location)

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

            if self.renderpath and n % self.save_weight_freq == self.save_weight_freq - 1:
                self.render_model(n + 1)


            update = str(total_episode_reward) + '\n'
            self.update_log(self.reward_logname, update)
            self.log_reward(ep_rew_dict)

            # store this episode's computation time and training reward history
            self.training_reward.append(total_episode_reward)

        update = 'q-learning algorithm complete'
        self.update_log(self.logname, update + '\n')
        print(update)

# import asyncio
#
# class LongTermMemory:
#
#     class _DatumFlag:
#         # binds to data to mark as processed or unprocessed and also which epoch it belongs to
#         def __init__(self, data, link):
#             self.data = data
#             self.processed = False
#             self.link = link # link to the processed/unprocessed _DatumFlag
#
#         def _link(self, other): #convenience func
#             self.link = other
#             other.link = self
#
#         def _toggleproc(self):
#             self.processed = not self.processed
#
#         def __eq__(self, other):
#             try:
#                 return self.data == other.data
#             except AttributeError:
#                 return self.data == other
#
#     def __init__(self, parent, model, len):
#
#         self.unprocessed = []
#         self.memory = []
#         self._working = False
#         self.model = model
#         self.parent = parent # the learner
#         self.len = len
#
#     def update_memory(self, episode_data):
#         # clip memory if it gets too long
#         num_episodes = len(self.unprocessed)
#         if num_episodes >= self.len:
#             num_delete = num_episodes - self.len
#             to_delete = self.unprocessed[:num_delete]
#             for data in to_delete: #data is a _datumflag
#                 if data.processed:
#                     self.memory.remove(data)
#                 self.unprocessed.remove(data)
#
#         # add most recent trial data to memory
#         self.unprocessed.append(self._DatumFlag(episode_data))
#         if not self._working:
#             self._working = True
#
#     async def _process(self, datum):
#         if not datum.processed:
#             data = datum.link
#             state, piece, locat = data[0]
#             next_state, next_piece, next_locat = data[1]
#             action = data[2]
#             reward = data[3]
#             done = data[4]
#
#             q = reward
#
#             # preprocess q using target network
#             if not done:
#                 next_state = torch.tensor(next_state).to(self.parent.device)
#                 next_piece = torch.tensor(next_piece).to(self.parent.device)
#                 next_locat = torch.tensor(next_locat).to(self.parent.device)
#
#                 qs = self.parent.target_network(next_state, next_piece, next_locat) #should be target
#                 q += self.parent.gamma * torch.max(qs)
#
#             state = torch.tensor(state).to(self.parent.device)
#             piece = torch.tensor(piece).to(self.parent.device)
#             locat = torch.tensor(locat).to(self.parent.device)
#
#             q_update = self.parent.target_network(state, piece, locat).squeeze(0) # should be target
#             q_update[action] = q
#
#             processed = q_update.detach().cpu().numpy()
#             state = state.cpu().squeeze(0).numpy()
#             piece = piece.float().squeeze(0).cpu().numpy()
#             locat = locat.float().cpu().numpy()
#
#     def __getitem__(self, item):
#         dflag = self.unprocessed[item]
#         if not dflag.processed:
#             dflag = await self._process(dflag)
#         data = dflag.link.data
#         return data



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
                t = torch.tensor(lst[item]).float()
            else:
                t = lst[item].float()
            t = t.to(self.device)
            out.append(t)
        return out