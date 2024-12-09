import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shutil
import os
import numpy as np
import argparse
from glob import glob
from utils import load, make_dir

fontsize = 12
nice_fonts = {
    "font.family": "serif",
    "axes.labelsize": fontsize,
    "font.size": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
}
matplotlib.rcParams.update(nice_fonts)
colors = ['r', 'b', 'k', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phys', type=str, default='tsp', help='The physics system to be studied')
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate for gd')
    parser.add_argument('--dim', type=int, default=20, help='The dimensionality of the data')
    parser.add_argument('--sigma', type=float, default=0.1, help='The standard deviation of the Gaussian noise')
    parser.add_argument('--x_lim', type=int, default=-1, help='The x-axis limit for the plots')
    return parser.parse_args()

def setup_directories(args):
    folder_path = f'exp/{args.phys}/lr_{args.lr}/dim_{args.dim}/sigma_{args.sigma}'
    plot_path = f'plots/{args.phys}/lr_{args.lr}/dim_{args.dim}/sigma_{args.sigma}'
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)
        print(f"Removed existing directory: {plot_path}")
    make_dir(plot_path)
    return folder_path, plot_path

def load_experiments(folder_path):
    num_exp = len(glob(f'{folder_path}/exp_*'))
    return [f'exp_{i}' for i in range(num_exp)]

def process_data(dataset, data_type):
    return np.array(dataset[f'function_values_{data_type}'])

def collect_function_values(folder_path, experiments):
    data_lists = {
        'gd': [],
        'rcd': [],
        'rcd_batch': [],
        'bcd_opt_rcd': [],
        'bcd_opt_rcd2': [],
        'bcd_c': [],
        'bcd_g': [],
        'bcd_reg': [],
        'bcd_random_thetas': [],
        'bcd_robust': [],
        'oicd': []
    }
    
    for experiment in experiments:
        file_path = os.path.join(folder_path, experiment, 'data_dict.pkl')
        data = load(file_path)
        for key in data.keys():
            data[key] = np.array(data[key])
        
        for key in list(data_lists.keys()):
            if f'function_values_{key}' in data:
                data_lists[key].append(process_data(data, key))
            else:
                data_lists.pop(key)
    
    return {key: pd.DataFrame(values) for key, values in data_lists.items()}

def calculate_mean_std(dataframes):
    stats = {}
    for key, df in dataframes.items():
        stats[key] = {
            'mean': np.array(df.mean(), dtype=float),
            'std': np.array(df.std(), dtype=float)
        }
    return stats

def plot_energy_ratio(stats, plot_path, x_lim):
    plt.figure()
    for key, color in zip(stats.keys(), colors[:len(stats)]):
        plt.plot(stats[key]['mean'], color=color, label=key.upper())
        plt.fill_between(
            range(len(stats[key]['mean'])), 
            stats[key]['mean'] - stats[key]['std'], 
            stats[key]['mean'] + stats[key]['std'], 
            color=color, alpha=0.2
        )
    plt.title('Energy Ratio')
    plt.xlabel('Iterations')
    plt.ylabel('Energy ratio: $E / E_{GS}$')
    plt.legend()
    if x_lim > 0:
        plt.xlim(0, x_lim)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{plot_path}/energy_HM.png')
    plt.clf()

def plot_function_evaluations(stats, plot_path, args, data):
    plt.figure()
    times = {}
    if 'gd' in stats:
        times['gd'] = np.arange(len(stats['gd']['mean'])) * args.dim * 2
    if 'rcd' in stats:
        times['rcd'] = np.arange(len(stats['rcd']['mean'])) * 2
    if 'rcd_batch' in stats:
        times['rcd_batch'] = np.arange(len(stats['rcd_batch']['mean'])) * args.dim * 2
    if 'bcd_opt_rcd' in stats:
        times['bcd_opt_rcd'] = np.arange(len(stats['bcd_opt_rcd']['mean'])) * 3
    if 'bcd_opt_rcd2' in stats:
        times['bcd_opt_rcd2'] = np.arange(len(stats['bcd_opt_rcd2']['mean'])) * 3
    if 'bcd_c' in stats:
        times['bcd_c'] = np.arange(len(stats['bcd_c']['mean'])) * 3
    if 'bcd_g' in stats:
        times['bcd_g'] = np.arange(len(stats['bcd_g']['mean'])) * 3
    if 'bcd_robust' in stats:
        times['bcd_robust'] = np.arange(len(stats['bcd_robust']['mean'])) * 3
    if 'bcd_reg' in stats:
        times['bcd_reg'] = np.arange(len(stats['bcd_reg']['mean'])) * data['fevl_num_each_iter_reg']
    if 'bcd_random_thetas' in stats:
        times['bcd_random_thetas'] = np.arange(len(stats['bcd_random_thetas']['mean'])) * 3
    if 'oicd' in stats:
        times['oicd'] = np.arange(len(stats['oicd']['mean'])) * 2


    for key, color in zip(times.keys(), colors[:len(times)]):
        plt.plot(times[key], stats[key]['mean'], color=color, label=key.upper())
        plt.fill_between(
            times[key], 
            stats[key]['mean'] - stats[key]['std'], 
            stats[key]['mean'] + stats[key]['std'], 
            color=color, alpha=0.2
        )

    plt.title('Energy Ratio')
    plt.xlabel('Number of function evaluations')
    plt.ylabel('Energy ratio: $ E / E_{GS}$')
    plt.legend()
    if args.x_lim > 0:
        plt.xlim(0, args.x_lim)
    plt.xlim(0, min(max(times[key]) for key in times.keys()))
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{plot_path}/energy_HM_fun_evals.png')
    plt.clf()

def main():
    args = parse_arguments()
    folder_path, plot_path = setup_directories(args)
    experiments = load_experiments(folder_path)
    dataframes = collect_function_values(folder_path, experiments)
    stats = calculate_mean_std(dataframes)
    plot_energy_ratio(stats, plot_path, args.x_lim)
    sample_data = load(os.path.join(folder_path, experiments[0], 'data_dict.pkl'))
    plot_function_evaluations(stats, plot_path, args, sample_data)

if __name__ == "__main__":
    main()
