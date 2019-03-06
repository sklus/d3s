"""
The goal of this file is to apply Koopman analysis techniques to ATARI games to see if anything useful
can be leveraged from koopman theory with regards to playing these games well.
"""
from matplotlib import pyplot as plt
from gym.envs import atari
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

env = atari.AtariEnv(game='pong', obs_type='image')
action_space = env.action_space

m = 100 # number of samples
# perform random actions in the environment and observe what happens
initial_image_rgb = env.reset()
initial_image_grey = rgb2gray(initial_image_rgb)
image_shape = initial_image_grey.shape
x0 = initial_image_grey.reshape(-1, 1) # make into vector
snapshots = np.zeros((len(x0), m), dtype=np.uint8)
snapshots[:, 0] = np.squeeze(x0) # assign initial observation vector as first column in snapshot mat
U = np.zeros((1, m-1))
for i in range(m-1):
    rand_action = action_space.sample()
    U[0,i] = rand_action
    image_rgb, reward, terminal, _ = env.step(rand_action)
    image_grey = rgb2gray(image_rgb)
    snapshots[:, i+1] = np.squeeze(image_grey.reshape(-1, 1))

# visualize to make sure we've done this correctly
def test_snapshot_recording_visualization(snapshots):
    plt.ion()
    for i in range(m):
        image_i = snapshots[:, i].reshape(image_shape)
        plt.imshow(image_i)
        plt.pause(0.01)
        plt.draw()

# now perform DMD on this shit
from d3s.algorithms import dmdc
X = snapshots[:, :m-1]
Y = snapshots[:, 1:m]
A, B, Phi = dmdc(X, Y, U, svThresh=1e-3)
from pprint import pprint
pprint(A)
print("\n")
pprint(B)

def mode_generator():
    for i in range(len(Phi[1])):
        mode_i = np.real(Phi[:, i]).reshape(image_shape)
        yield mode_i

def mode_visualizer(mode_generator):
    try:
        img = next(mode_generator)
        plt.figure()
        plt.imshow(img)
    except:
        pass

f = mode_generator()
mode_visualizer(f)
