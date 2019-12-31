import shapes3d as s3d
import numpy as np
from itertools import product
import copy
import render
import time

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
        self.done = False

        self.score = 0
        self.cumscore = 0
        self.total_pieces = 0

        self.packing_percentage = 0
        self.unreachable_percentage = 0

        self.start_time = time.time()

        # after some ticks, we force a drop
        self.drop_tick_counter = 0

        # ##################################
        # Explicitly machine learning stuff:
        # ##################################
        if rewards:
            '''
                The rewards scheme is as follows:
                    for clearing n lines consecutively (0 <= n <= 4): 
                    r = clear_reward * n ** clear_exponent
                    
                    for dropping the piece one unit:
                    r = drop_reward
                    
                    for doing any action other than a drop:
                    r = null_reward
                    
                    every tick, spacial efficiency is computed:
                    r = compute_spacial_efficiency() * packing_reward
            '''

            self.clear_reward = rewards[0]
            self.clear_exponent = rewards[1]
            self.pieces_reward = rewards[2]
            self.packing_penalty = rewards[3]
            self.holes_reward = rewards[4]
            self.game_over_penalty = rewards[5]

        self.action_space = [Action('drop')] + [Action('translate', i, j) for i, j in product(range(2), (-1, 1))] + \
                            [Action('rotate', i, j) for i, j in product(range(3), (-1, 1))]

        # Input/output in the sense of a perceptron
        self.output_dimension = len(self.action_space)

        self.board_size = 1
        for i in board_shape:
            self.board_size *= i

        # 2 pieces (current/next), the board, and the current location
        self.input_dimension = 2 * self.current.total_dimension + self.board_size + 3

    def get_new_piece(self):
        shape = np.random.choice(list(s3d.shapelib.keys()))
        newpiece = s3d.Tetromino(s3d.shapelib[shape],
                                 [int(self.board_extents[0]/2), int(self.board_extents[0]/2), 0])
        return newpiece

    # returns percentage utilization of rows that currently have peices
    def compute_packing_efficiency(self):
        denom = 0
        sum = 0
        for z in range(self.board.shape[2]):
            if self.board[:, :, z].any():
                denom += self.board.shape[0] * self.board.shape[1]
                sum += np.sum(self.board[:, :, z])
        if denom != 0:
            return 1 - sum / denom
        else:
            return 0

    # we count the number of unreachable blocks, then divide by the number of placed blocks; we then store the score;
    # Since this operation can be expensive, we only want to perform it when the board changes, but want to use it
    # in our per-tick reward scheme
    def compute_unreachable_percentage(self):
        placed_blocks = np.sum(self.board)
        visited = set([])

        # compute the number of reachable blocks recursively
        def visit(curloc):
            for axis in range(3):
                if curloc[axis] not in range(self.board.shape[axis]):
                    return
            if self.board[tuple(curloc)]:
                return
            if tuple(curloc) not in visited:
                visited.add(tuple(curloc))
            for axis, direc in product(range(3), (-1, 1)):
                temp_loc = copy.copy(curloc)
                temp_loc[axis] += direc
                if tuple(temp_loc) not in visited:
                    visit(temp_loc)

        for x, y in product(range(self.board.shape[0], self.board.shape[1])):
            visit([x, y, 0])

        reachable_but_empty = len(visited)
        unreachable = self.board_size - (reachable_but_empty + placed_blocks)
        return unreachable / placed_blocks

    # adds the current piece to the board. If inplace, then directly modifies self.board, then returns self.board;
    # Otherwise, makes a copy, then returns the modified copy
    def add_to_board(self, inplace=True):
        if not inplace:
            temp = copy.deepcopy(self.board)
        else:
            temp = self.board
        for x, y, z in product(range(self.current.shape[0]), range(self.current.shape[1]),
                               range(self.current.shape[2])):
            if self.current.matrix[x, y, z] != 0:
                temp[x + self.current.location[0],
                     y + self.current.location[1],
                     z + self.current.location[2]] = self.current.color
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

    # must pass Action objects; transitions the board
    def update(self, action):

        self.score = 0

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

        if self.drop_tick_counter >= 5:
            drop()
            self.drop_tick_counter = 0

        # add the packing efficiency
        self.score += 0.8 * self.cumscore + self.packing_percentage * self.packing_penalty + (1 - self.unreachable_percentage) * \
                      self.holes_reward + self.total_pieces * self.pieces_reward

        # check the board
        if not self.is_valid_position(self.current.matrix, offset=(0, 0, 1)):
            self.board = self.add_to_board()
            self.cumscore += self.clear_reward * self.remove_complete_lines() ** self.clear_exponent
            self.current = self.next
            self.next = self.get_new_piece()
            self.total_pieces += 1
            self.unreachable_percentage = self.compute_unreachable_percentage()
            self.packing_percentage = self.compute_packing_efficiency()
            if not self.is_valid_position(self.current.matrix):
                self.done = True


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
        self.unreachable_percentage = 0
        self.packing_percentage = 0

        self.start_time = time.time()
        return self

    # just the attributes(? can't think of the correct word right now) we need for deep learning
    # if flat, returns as a 1-D vector
    def stripped(self, flat=True):
        stripped = [self.current.matrix, self.current.location, self.next.matrix, self.board]
        if flat:
            stripped = flatten_state(stripped)
        return stripped

    # This is the simulator function Q-Learner will call (implementing this as a __call__ method
    # cleans up the syntax quite a bit)
    def __call__(self, action, verbose=False):
        if verbose:
            print(action)
        self.update(action)
        return self, self.score, self.done

    def render_frame(self):
        draw_mat = self.add_to_board(inplace=False)
        to_draw = np.flip(draw_mat, 2)
        return render.generate_external_faces(to_draw)

    def knitted_board(self):
        if not self.done:
            return np.flip(self.add_to_board(inplace=False), 2)
        else:
            return np.flip(self.board, 2)


# flattens the state into a 1D numpy array
def flatten_state(state):
    arrays = [s.flatten() for s in state]
    out = np.concatenate(arrays).ravel()
    return out


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