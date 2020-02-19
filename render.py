import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shapes3d as s3d
from itertools import product, combinations
from mpl_toolkits.mplot3d import art3d
import time
import tetris3d as sw
from matplotlib.animation import FuncAnimation, FFMpegWriter, writers
import os
import pickle
import torch

'''
    Tools for rendering and animating tetris-like numpy matrices
'''


# courtesy user "karlo" on stack overflow
# .../questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# render a numpy array as a wireframe
def render_wireframe(board_array, ax_to_plot_on):

    shape = board_array.shape

    # draws a cube at the specificed x,y,z location with the given color
    def draw_cube(offset, color):
        r = [0, 1]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s - e)) == r[1] - r[0]:
                s = np.add(s, offset)
                e = np.add(e, offset)
                ax_to_plot_on.plot3D(*zip(s, e), color=color)

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if board_array[x, y, z] != 0:
                    draw_cube(np.array([x, y, z]), s3d.color_vec[board_array[x, y, z]])


# returns a list of Poly3D collections, so need to use a forloop to unlist
def generate_external_faces(board_array, verbose=False):
    start_time = time.time()
    faces_drawn = 0

    shape = board_array.shape
    collection = []

    def generate_face(location, color1, axis1, direction1):
        x_face = [[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 1],
                  [0, 1, 0]]

        y_face = [[0, 0, 0],
                  [1, 0, 0],
                  [1, 0, 1],
                  [0, 0, 1]]

        z_face = [[0, 0, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [0, 1, 0]]

        faces = [x_face, y_face, z_face]

        face = faces[axis1]
        for row in face:
            if direction1 == 1:
                row[axis1] += 1
            for i in range(3):
                row[i] += location[i]

        face_nump = np.array(face)
        side = art3d.Poly3DCollection([face_nump])
        side.set_edgecolor('k')
        side.set_facecolor(color1)
        collection.append(side)
        if verbose:
            axes = ['x', 'y', 'z']
            print("Gnerated face at %s, along %s axis" % (location, axes[axis1]))

    # iterate through the array and check to see if neighbors. If no neighbor, generate face
    for x, y, z in product(range(shape[0]), range(shape[1]), range(shape[2])):
        if board_array[x, y, z] != 0:
            color = s3d.color_vec[int(board_array[x, y, z])]

            for axis, direction in product(range(3), [-1,1]):
                temp = [x, y, z]
                if 0 <= temp[axis] + direction < shape[axis]:
                    temp[axis] += direction
                    if board_array[tuple(temp)] == 0:
                        generate_face([x, y, z], color, axis, direction)
                        faces_drawn += 1

                else:  # its an edge face, so draw it
                    generate_face([x, y, z], color, axis, direction)
                    faces_drawn += 1

    if verbose:
        print("=========================================\n"
              "FACES DRAWN: %s \nTIME ELAPSED: %s seconds" %
              (faces_drawn, time.time() - start_time))
    return collection


# the aforementioned forloop to unlist the collection
def plot_collections(col_list, ax_to_plot_on):
    for i in col_list:
        ax_to_plot_on.add_collection3d(i)


# given list of game states, animate
def animate_from_queue(states_queue, fig_to_plot, ax_to_plot_on, framedelay=10):

    def draw_frame(frame):
        [p.remove() for p in reversed(ax_to_plot_on.collections)]
        plot_collections(generate_external_faces(frame), ax_to_plot_on)

    anim = FuncAnimation(fig_to_plot, draw_frame, frames=states_queue, interval=framedelay, repeat=False)
    return anim


# moving average function
def moving_average(x, D):
    y = []
    for p in range(len(x) + 1):
        # make next element
        b = np.sum(x[np.maximum(0, p - D):p]) / float(D)
        y.append(b)
    return np.array(y)


def get_reward_history(logname, **kwargs):
    start = 1
    window_length = 5
    if 'window_length' in kwargs:
        window_length = kwargs['window_length']
    if 'start' in kwargs:
        start = kwargs['start']

    # load in total episode reward history
    data = np.loadtxt(logname)
    ave = moving_average(data, window_length)

    return data, ave

def plot_reward_history(logname, **kwargs):

    start = 1
    window_length = 5
    if 'window_length' in kwargs:
        window_length = kwargs['window_length']
    if 'start' in kwargs:
        start = kwargs['start']

    data, ave = get_reward_history(logname, **kwargs)
    # create figure
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # plot total reward history
    ax1.plot(data)
    ax1.set_xlabel('episode', labelpad=8, fontsize=13)
    ax1.set_ylabel('total reward', fontsize=13)

    ave[:window_length] = np.nan
    ax2.plot(ave, linewidth=3)
    ax2.set_xlabel('episode', labelpad=8, fontsize=13)
    ax2.set_ylabel('ave total reward', fontsize=13)

CPU = torch.device('cpu')


def make_torch(array, device=CPU):
    tens = torch.from_numpy(array.copy())
    tens = tens.float()
    tens = tens.unsqueeze(0)
    tens = tens.unsqueeze(0)
    return tens.to(device)

def render_from_model(model, fig, ax, tetris_instance, save_path=None, device=CPU):

    def renormalize_vec(tensor, idx):
        tensor = tensor.squeeze(0)
        tensor[idx] = 0
        sum = torch.sum(tensor)
        return tensor / sum

    frames_queue1 = []
    frame_count = 0
    while not tetris_instance.done:
        frame_count += 1

        state = make_torch(tetris_instance.board, device)
        piece = make_torch(tetris_instance.current.matrix, device)
        location = torch.from_numpy(tetris_instance.current.location).float().unsqueeze(0).to(device)

        output = model(state, piece, location)
        invalid_actions = []
        for i in range(len(tetris_instance.action_space)):  # todo paralellize
            invalid_actions.append(not tetris_instance.is_valid_action(tetris_instance.action_space[i]))
        invalid_actions = np.array(invalid_actions)
        output = renormalize_vec(output, invalid_actions)

        action = tetris_instance.action_space[torch.argmax(output)]
        tetris_instance(action)
        if frame_count % 3 == 0:
            frames_queue1.append(tetris_instance.knitted_board())

    animation = animate_from_queue(frames_queue1, fig, ax, framedelay=20)
    if save_path:
        animation.save(save_path, writer='imagemagick', fps=30)
    return animation


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    # tetris settings
    clear_reward = 1000
    pieces_reward = 0.5
    packing_reward = 0.05
    reachable_penalty = 0.001
    variance_penalty = 0.001
    translation_reward = 0.0001
    rotation_reward = 0.00009
    game_over_penalty = 50
    game_len_reward = 0.0001

    rewards = {
        'clear_reward': clear_reward,
        'pieces_reward': pieces_reward,
        'packing_reward': packing_reward,
        'reachable_penalty': reachable_penalty,
        'variance_penalty': variance_penalty,
        'game_over_penalty': game_over_penalty,
        'translation_reward': translation_reward,
        'rotation_reward': rotation_reward,
        'game_len_reward': game_len_reward
    }

    import torch
    from model import DenseNet

    save_name = 'oleg_tuned'
    exit_window = 100

    # an example of the animation renderer
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    axis_extents = (7, 7, 15)
    ax.set_xlim3d(0, 7)
    ax.set_ylim3d(0, 7)
    ax.set_zlim3d(0, 15)

    N=128

    game = sw.GameState(board_shape=axis_extents, rewards=rewards)

    model = DenseNet(11, in_channels=N, neurons=256, fc_channels=N, dense_channels=N,
                     concat_channels=N, pool=7)

    model.to(torch.device('cpu'))

    model.load_state_dict(torch.load('saved_model_weights/219-oleg-continued/219-oleg-continued300.pth'))
    model.eval()

    render_from_model(model, fig, ax, game, save_path='images/219-oleg-500.gif')


