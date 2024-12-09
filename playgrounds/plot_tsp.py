# plot the data generated from `run_factor.py`


import pickle
import matplotlib.pyplot as plt
from sysflow.utils.common_utils.file_utils import make_dir, load
from glob import glob
import numpy as np

def load_data(num_layer, learning_rate_rcd, learning_rate_gd):
    rcd_path = f"results/tsp/layer_{num_layer}/lr_{learning_rate_rcd}/rcd.pkl"
    gd_path = f"results/tsp/layer_{num_layer}/lr_{learning_rate_gd}/gd.pkl"

    with open(rcd_path, 'rb') as rcd_file:
        rcd_data = pickle.load(rcd_file)['f']
    with open(gd_path, 'rb') as gd_file:
        gd_data = pickle.load(gd_file)['f']

    return rcd_data, gd_data

def plot_data(rcd_data, gd_data, plot_name, plot_type):
    plt.figure(figsize=(10, 6))

    # assuming the data is in a format that can be directly plotted
    plt.plot(rcd_data, label='RCD')
    if plot_type == 'partial':
        plt.plot( 90 * np.arange(len(gd_data)), gd_data, label='GD')
    else: 
        plt.plot(gd_data, label='GD')

    if plot_type == 'partial':
        plt.xlabel('Number of partial derivative evaluations')
    else:
        plt.xlabel('Number of steps')
    plt.xlim(0, 4000)
    plt.ylabel('Value')
    plt.title('Comparison between RCD and GD')
    plt.grid()
    plt.legend()

    plt.savefig(plot_name)
    plt.clf()
    plt.close()

# specify the number of layers and the learning rate
num_layer = 5
learning_rate = 0.10000000149011612

path = glob('results/tsp/layer_*')
num_layer_list = [10]
for num_layer in num_layer_list: 
    
    path = glob(f'results/tsp/layer_{num_layer}/lr_*')
    learning_rate_list = [p.split('_')[-1] for p in path]

    learning_rate_rcd = 1e-3
    learning_rate_gd = 1e-4

    # load the data
    rcd_data, gd_data = load_data(num_layer, learning_rate_rcd, learning_rate_gd)

    make_dir(f'plots/factor/tsp/layer_{num_layer}')
    plot_name = f'plots/factor/tsp/layer_{num_layer}/lr_{learning_rate_gd}.png'

    # plot the data
    plot_data(rcd_data, gd_data, plot_name, 'iter')

    make_dir(f'plots/factor/tsp/layer_{num_layer}')
    plot_name = f'plots/factor/tsp/layer_{num_layer}/lr_{learning_rate_gd}_partial.png'

    # plot the data
    plot_data(rcd_data, gd_data, plot_name, 'partial')