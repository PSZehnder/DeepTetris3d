import shapes3d as s3d
import numpy as np
from itertools import product
import copy
import render
import time
from math import sqrt

'''
    A 3D tetris clone based on Al Sweigart's example code. Code has been modified in order to:
        1) Bring the game into 3D
        2) Be compatible with the renderer I wrote
        3) Have an interface similar to the OpenAI gym for Q-Learning purposes
    
    The original code can be found here: https://inventwithpython.com/pygame/chapter7.html
'''

class GameState:

    def __init__(self, board_shape, rewards=None):
        self.board_extents = board_shape
        self.current = self.get_new_piece()
        self.next = self.get_new_piece()
        self.board = np.zeros(tuple(self.board_extents))
        self.board_with_piece = np.copy(self.board)
        self.done = False

        self.board_size = board_shape[0] * board_shape[1] * board_shape[2]

        self.score = 0
        self.cumscore = 0
        self.total_pieces = 0

        self.start_time = time.time()

        # after some ticks, we force a drop
        self.drop_tick_counter = 0
        self.ticks = 0

        if rewards:
            self.rewards = rewards

            self.clear_reward = 0
            self.pieces_reward = 0
            self.packing_reward = 0
            self.variance_penalty = 0
            self.translation_reward = 0
            self.rotation_reward = 0
            self.game_over_penalty = 0
            self.game_len_reward = 0
            self.height_penalty = 0
            self.empty_column_penalty = 0
            self.overhang_penalty = 0

            for k, v in rewards.items():
                self.__setattr__(k, v)

            # used to tabulate some metrics
            self.unreachable = None
            self.previous_piece_count = self.total_pieces

        self.action_space = [Action('drop')] + [Action('translate', i, j) for i, j in product(range(2), (-1, 1))] + \
                            [Action('rotate', i, j) for i, j in product(range(3), (-1, 1))]

        # action space cardinality
        self.output_dimension = len(self.action_space)

    def reward(self, n=0, calc=False, done=False):

        clear_reward = self.clear_reward * n ** 2
        piece_reward = (int(self.total_pieces > self.previous_piece_count) * self.pieces_reward)
        game_over_penalty = int(done) * (-self.game_over_penalty)
        tick_reward =  self.game_len_reward * self.total_pieces ** 1.25
        packing_efficiency = (self.compute_packing_efficiency() * self.packing_reward)
        variance_penalty = (self.compute_columnar_variance() * self.variance_penalty)
        height_penalty = self.height_penalty * self.current.location[2] ** 1.25
        empty_column_penalty = self.empty_columns() * self.empty_column_penalty
        self.previous_piece_count = self.total_pieces

        result_dict = {
            'clear_reward' : clear_reward,
            'piece_reward' : piece_reward,
            'game_over_penalty': game_over_penalty,
            'packing_efficiency': packing_efficiency,
            'columnar_variance': variance_penalty,
            'empty_column_penalty': empty_column_penalty,
            'height_penalty': height_penalty,
            'tick_reward': tick_reward,
        }
        if not calc:
            for k in result_dict.keys():
                result_dict[k] = 0

        if done:
            for k in result_dict.keys():
                result_dict[k] = 0
            result_dict['game_over_penalty'] = game_over_penalty

        result_dict['total'] = sum(list(result_dict.values()))

        return result_dict

    def height(self):
        for z in range(self.board.shape[2]):
            if self.board[:, :, z].any():
                return self.board.shape[2] - z
        return 0

    def empty_columns(self):
        denom = self.board.shape[0] * self.board.shape[1]
        numer = 0
        for x in range(self.board.shape[0]):
            for y in range(self.board.shape[1]):
                numer += int(self.board[x, y, self.board.shape[2] - 1])
        return numer / denom

    def get_new_piece(self):
        shape = np.random.choice(list(s3d.shapelib.keys()))
        location = (self.board_extents[0] // 2, self.board_extents[1] // 2, 0)
        newpiece = s3d.Tetromino(s3d.shapelib[shape], location)
        return newpiece

    def compute_columnar_variance(self):
        summed_columns = np.sum(self.board, axis=2)
        variance1 = np.var(np.array([np.var(i) for i in summed_columns]))
        variance2 = np.var(np.array([np.var(i) for i in summed_columns.T]))
        if variance1 + variance2 > 0:
            return 2 / (variance1 + variance2)
        else:
            return 0
        # total_mass = np.sum(self.board)
        # ideal_avgmass = total_mass / (self.board_extents[0] * self.board_extents[1])
        #
        # ideal = np.ones((self.board_extents[0], self.board_extents[1])) * ideal_avgmass
        # diff = np.absolute(ideal - summed_columns)
        # diff = np.square(diff)
        # return np.sum(diff) / (self.board_extents[0] * self.board_extents[1])

    # returns percentage utilization of rows that currently have peices
    def compute_packing_efficiency(self): #todo: fix this feature
        denom = 0
        sum = 0
        for z in range(self.board.shape[2]):
            denom += self.board.shape[0] * self.board.shape[1] * z
            sum += np.sum(self.board[:, :, z]) * z
        if denom != 0:
            height = self.height()
            height = max(1, height)
            return sum / denom / height
        else:
            return 0

    def count_over_hangs(self):
        count = 0
        for x, y, z in product(range(self.board_extents[0]),
                               range(self.board_extents[1]), range(1, self.board_extents[2])):
            if self.board[x, y, z] == 0 and self.board[x, y, z - 1] == 1:
                count += 1
        return count

    # adds the current piece to the board. If inplace, then directly modifies self.board, then returns self.board;
    # Otherwise, makes a copy, then returns the modified copy. by defualt, set to 1
    def add_to_board(self, inplace=True, color=None):
        if not inplace:
            temp = copy.deepcopy(self.board)
        else:
            temp = self.board
        if not color:
            color = self.current.color
        for x, y, z in product(range(self.current.shape[0]), range(self.current.shape[1]),
                               range(self.current.shape[2])):
            if self.current.matrix[x, y, z] != 0:
                temp[x + self.current.location[0],
                     y + self.current.location[1],
                     z + self.current.location[2]] = color
        return temp

    # simple bounds check for a given location (tuple)
    def is_on_board(self, loc):
        return 0 <= loc[0] < self.board_extents[0] and \
               0 <= loc[1] < self.board_extents[1] and loc[2] < self.board_extents[2]

    # checks if the piece with a given offset is within the bound and not colliding
    def is_valid_position(self, newmat, offset=(0, 0, 0)):
        for x, y, z in product(range(newmat.shape[0]), range(newmat.shape[1]), range(newmat.shape[2])):
            if newmat[x, y, z] == 0:
                continue
            adjloc = [x + self.current.location[0] + offset[0],
                      y + self.current.location[1] + offset[1],
                      z + self.current.location[2] + offset[2]]
            if not self.is_on_board(adjloc):
                return False
            if self.board[tuple(adjloc)] != 0:
                return False
        return True

    def is_complete_line(self, z):
        for x, y in product(range(self.board_extents[0]), range(self.board_extents[1])):
            if self.board[x, y, z] == 0:
                return False
        return True

    # remove the complete lines and drop the ones above it in the classic tetris fashion; return num consec lines
    # cleared all at once
    def remove_complete_lines(self):
        consec_clears = 0
        z = self.board_extents[2] - 1
        while z >= 0:
            if self.is_complete_line(z):
                for x, y, pulldownz in product(range(self.board_extents[0]),
                                               range(self.board_extents[1]), range(z, 0, -1)):
                    self.board[x, y, pulldownz] = self.board[x, y, pulldownz - 1]
                for x, y in product(self.board_extents[0], self.board_extents[1]):
                    self.board[x, y, 0] = 0
                consec_clears += 1
            else:
                z -= 1
        return consec_clears

    def is_valid_action(self, action):

        def check_translate(axis, direction):
            offset = [0, 0, 0]
            offset[axis] = direction
            temp = copy.copy(self.current.location)
            temp[axis] += direction

            return self.is_valid_position(self.current.matrix, offset)

        def check_rotate(axis, direction=None):
            axes = [0, 1, 2]
            del axes[axis]
            temp = np.rot90(self.current.matrix, k=direction, axes=axes)
            return self.is_valid_position(temp)

        def check_drop(axis=None, direction=None):
            return True

        switcher = {
            'translate': check_translate,
            'rotate': check_rotate,
            'drop': check_drop
        }

        return switcher[action.type](action.axis, action.direction)

    # must pass Action objects; transitions the board
    def update(self, action):

        self.score = 0
        self.ticks += 1

        # action block: each takes an axis and direction
        def translate(axis, direction):
            offset = [0, 0, 0]
            offset[axis] = direction
            temp = copy.copy(self.current.location)
            temp[axis] += direction

            if self.is_valid_position(self.current.matrix, offset):
                self.current.location = temp

        def rotate(axis, direction):
            axes = [0, 1, 2]
            del axes[axis]
            temp = np.rot90(self.current.matrix, k=direction, axes=axes)
            if self.is_valid_position(temp):
                self.current.matrix = temp
                self.current.shape = temp.shape

        def drop(axis=None, direction=None):
            if self.is_valid_position(self.current.matrix, offset=(0, 0, 1)):
                self.current.location[2] += 1

        switcher = {
            'translate': translate,
            'rotate': rotate,
            'drop': drop
        }

        switcher[action.type](action.axis, action.direction)

        self.drop_tick_counter += 1

        if self.drop_tick_counter >= 2:
            drop()
            self.drop_tick_counter = 0

        self.board_with_piece = self.add_to_board(inplace=False, color=100)

        self.tick_reward = self.reward()

        # check the board
        if not self.is_valid_position(self.current.matrix, offset=(0, 0, 1)):
            self.board = self.add_to_board(color=1)
            removed_lines = self.remove_complete_lines()
            self.tick_reward = self.reward(n=removed_lines, calc=True)
            self.cumscore += self.tick_reward['total']
            self.current = self.next
            self.next = self.get_new_piece()
            self.total_pieces += 1
            if not self.is_valid_position(self.current.matrix):
                self.done = True
                self.tick_reward = self.reward(removed_lines, calc=True, done=self.done)

    # returns a clean version of the game state with the same parameters
    def reset(self):
        self.current = self.get_new_piece()
        self.next = self.get_new_piece()
        self.board = np.zeros(tuple(self.board_extents))
        self.done = False

        self.score = 0
        self.cumscore = 0
        self.total_pieces = 0

        self.drop_tick_counter = 0
        self.ticks = 0
        self.unreachable = None

        self.start_time = time.time()
        return self

    # This is the simulator function Q-Learner will call (implementing this as a __call__ method
    # cleans up the syntax quite a bit)
    def __call__(self, action, verbose=False):
        if verbose:
            print(action)
        self.update(action)
        return self, self.tick_reward, self.done

    def render_frame(self):
        draw_mat = self.add_to_board(inplace=False)
        to_draw = np.flip(draw_mat, 2)
        return render.generate_external_faces(to_draw)

    def knitted_board(self):
        if not self.done:
            return np.flip(self.add_to_board(inplace=False), 2)
        else:
            return np.flip(self.board, 2)


# Creating an explicit class for the actions is probably unnecessary silliness, but I mean
# if we need to do anything with them w.r.t the learnng part of this project It's here anyway.
# Plus, I get to practice "good coding" and the __repr__ method.
class Action:

    def __init__(self, typ, axis=None, direction=None):
        if axis:
            if axis > 2:
                raise Exception('Action: axis %d out of bounds' % axis)
        if direction:
            if direction != 1 and direction != -1:
                raise Exception('Action: direction %d out of bounds' % direction)

        self.axis = axis
        self.direction = direction
        self.type = typ

    def __repr__(self):
        return '%s (%s %s)' % (self.type, self.axis, self.direction)