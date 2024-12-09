# plot the data generated from `run_factor.py`


import pickle
import matplotlib.pyplot as plt
from utils import make_dir, load
from glob import glob
import numpy as np
class OptimizerLog:
    """Log to store optimizer's intermediate results"""
    def __init__(self):
        self.evaluations = []
        self.parameters = []
        self.costs = []
    def update(self, evaluation, parameter, cost, _stepsize):
        """Save intermediate results. Optimizer passes five values
        but we ignore the last two."""
        self.evaluations.append(evaluation)
        self.parameters.append(parameter)
        self.costs.append(cost)

def load_data(num_layer, learning_rate):
    rcd_path = f"results/quantum_classifier/lr_{learning_rate}/rcd.pkl"
    gd_path = f"results/quantum_classifier/lr_{learning_rate}/gd.pkl"

    with open(rcd_path, 'rb') as rcd_file:
        rcd_data = pickle.load(rcd_file)
        # result = rcd_data['result']
        # opt_var = result.x
        # opt_value = result.fun
        opt_log = rcd_data['log']
        rcd_data = opt_log.costs
        # eval = opt_log.evaluations

    with open(gd_path, 'rb') as gd_file:
        gd_data = pickle.load(gd_file)
        opt_log = gd_data['log']
        gd_data = opt_log.costs

    return rcd_data, gd_data

def plot_data(rcd_data, gd_data, plot_name, plot_type):
    plt.figure(figsize=(10, 6))

    # assuming the data is in a format that can be directly plotted
    plt.plot(rcd_data, label='RCD')
    if plot_type == 'partial':
        plt.plot(12 * np.arange(len(gd_data)), gd_data, label='GD')
    else: 
        plt.plot( gd_data, label='GD')

    if plot_type == 'partial':
        plt.xlabel('Number of partial derivative evaluations')
        plt.xlim(0, 2000)
    else: 
        plt.xlabel('Number of steps')
    plt.ylabel('Value')
    plt.title('Comparison between RCD and GD')
    plt.grid()
    plt.legend()

    plt.savefig(plot_name)
    plt.clf()
    plt.close()

# specify the number of layers and the learning rate
num_layer = 5
learning_rate = 0.07

    
path = glob(f'results/quantum_classifier/lr_*')
learning_rate_list = [p.split('_')[-1] for p in path]

for learning_rate in learning_rate_list:

    # load the data
    rcd_data, gd_data = load_data(num_layer, learning_rate)

    make_dir(f'plots/factor/quantum_classifier')
    plot_name = f'plots/factor/quantum_classifier/lr_{learning_rate}.png'

    # plot the data
    plot_data(rcd_data, gd_data, plot_name, 'iter')

    make_dir(f'plots/factor/quantum_classifier/partial')
    plot_name = f'plots/factor/quantum_classifier/partial/lr_{learning_rate}.png'
    plot_data(rcd_data, gd_data, plot_name, 'partial')
    
