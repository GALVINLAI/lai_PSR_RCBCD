# plot the data generated from `run_factor.py`


import pickle
import matplotlib.pyplot as plt
from sysflow.utils.common_utils.file_utils import make_dir, load
from glob import glob
import numpy as np

def load_data(num_layer, learning_rate):
    rcd_path = f"results/maxcut/layer_{num_layer}/lr_{learning_rate}/rcd.pkl"
    gd_path = f"results/maxcut/layer_{num_layer}/lr_{learning_rate}/gd.pkl"

    with open(rcd_path, 'rb') as rcd_file:
        rcd_data = pickle.load(rcd_file)['f']
    with open(gd_path, 'rb') as gd_file:
        gd_data = pickle.load(gd_file)['f']

    return rcd_data, gd_data

def plot_data(rcd_data, gd_data, plot_name):
    plt.figure(figsize=(10, 6))

    # assuming the data is in a format that can be directly plotted
    plt.plot(rcd_data, label='RCD')
    plt.plot(gd_data, label='GD')

    plt.xlabel('Number of steps')
    plt.ylabel('Value')
    plt.title('Comparison between RCD and GD')
    plt.grid()
    plt.legend()

    plt.savefig(plot_name)
    plt.clf()
    plt.close()


def plot_data_partial(rcd_data, gd_data, plot_name):
    plt.figure(figsize=(10, 6))

    # assuming the data is in a format that can be directly plotted
    plt.plot(rcd_data, label='RCD')
    plt.plot(20 * np.arange(len(gd_data)), gd_data, label='GD')

    plt.xlabel('Number of partial derivative evaluations')
    plt.ylabel('Value')
    plt.title('Comparison between RCD and GD')
    plt.xlim(0, 1000)
    plt.grid()
    plt.legend()

    plt.savefig(plot_name)
    plt.clf()
    plt.close()
    
# specify the number of layers and the learning rate
num_layer = 5
learning_rate = 0.10000000149011612

path = glob('results/maxcut/layer_*')
num_layer_list = [int(p.split('_')[-1]) for p in path]
for num_layer in num_layer_list: 
    
    path = glob(f'results/maxcut/layer_{num_layer}/lr_*')
    learning_rate_list = [p.split('_')[-1] for p in path]
    
    for learning_rate in learning_rate_list:

        # load the data
        rcd_data, gd_data = load_data(num_layer, learning_rate)

        make_dir(f'plots/factor/maxcut/layer_{num_layer}')
        plot_name = f'plots/factor/maxcut/layer_{num_layer}/lr_{learning_rate}.png'

        # plot the data
        plot_data(rcd_data, gd_data, plot_name)

    for learning_rate in [1.2400000095367432]:
        # load the data
        rcd_data, gd_data = load_data(num_layer, learning_rate)

        make_dir(f'plots/factor/maxcut/layer_{num_layer}')
        plot_name = f'plots/factor/maxcut/layer_{num_layer}/maxcut_lr_1.24_partial.png'

        # plot the data
        plot_data_partial(rcd_data, gd_data, plot_name)