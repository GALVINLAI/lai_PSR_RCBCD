import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shutil

'''
This code snippet is used to process and visualize the performance data of optimization algorithms 
(Gradient Descent and Random Coordinate Descent) on physical systems (such as the Transverse Field Ising Model, TFIM). 
Specifically, the code loads information from a set of experimental data, calculates some statistics, 
and generates multiple charts to display the results. Below is a detailed explanation.
'''

fontsize = 20
# Set the font and style parameters for plotting to ensure consistent style in the document
nice_fonts = {
# Use LaTeX to write all text
# "text.usetex": True,
"font.family": "serif",
# Use 10pt font in plots, to match 10pt font in document
"axes.labelsize": fontsize,
"font.size": fontsize,
# Make the legend/label fonts a little smaller
"legend.fontsize": fontsize,
"xtick.labelsize": fontsize,
"ytick.labelsize": fontsize,
}
matplotlib.rcParams.update(nice_fonts)

import os
from utils import load, make_dir
import numpy as np
import argparse
from glob import glob

# Initialize ArgumentParser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--phys', type=str, default='tsp', 
                    help='The physics system to be studied')

parser.add_argument('--lr', type=float, default=0.0001, 
                    help='The learning rate for gd')

parser.add_argument('--dim', type=int, default=20, 
                    help='The dimensionality of the data')

parser.add_argument('--sigma', type=float, default=0.0, 
                    help='The standard deviation of the Gaussian noise')

parser.add_argument('--x_lim', type=int, default= -1,
                    help='The x-axis limit for the plots')
# Parse arguments
args = parser.parse_args()

# Construct the folder path from the parsed arguments
folder_path = f'exp/{args.phys}/lr_{args.lr}/dim_{args.dim}/sigma_{args.sigma}'
plot_path = f'plots/{args.phys}/lr_{args.lr}/dim_{args.dim}/sigma_{args.sigma}'

# make_dir is custom
# If the directory already exists, it will not raise an error but simply ignore this operation.
# Note: We do not recommend using it, because if old plots exist, it does nothing.
# Check if the folder exists and remove it if it does.
if os.path.exists(plot_path):
    shutil.rmtree(plot_path)
    print(f"Removed existing directory: {plot_path}")
make_dir(plot_path)

# list of experiments to load

# glob is a function in the Python standard library, located in the glob module, used to find all pathnames matching a specific pattern.
# The glob function returns a list of filenames that match the given pattern.
# The pattern is f'{folder_path}/exp_*',
# This means we look for all files or folders starting with exp_ in the folder_path directory.
num_exp = len(glob(f'{folder_path}/exp_*'))

# This line generates a list of experiment names. Suppose num_exp is 5, then:
# ['exp_0', 'exp_1', 'exp_2', 'exp_3', 'exp_4']
experiments = [f'exp_{i}' for i in range(num_exp)]

def process_data(dataset, data_type):
    # Define the process_data function to process and calculate the function values, eigenvalues, and Lipschitz constants ratios
    # for the given data type ('gd' or 'rcd').

    # Add function_values to the list
    function_values = np.array(dataset[f'function_values_{data_type}'])

    # Process eigen_values
    eigen_values = np.array([np.max(np.abs(x)) for x in dataset[f"eigen_values_{data_type}"]])
    
    # Process lipshitz_diagonal_values
    lipshitz_diagonal_values = np.array([np.mean(np.abs(x)) for x in dataset[f"lip_diag_values_{data_type}"]])
    max_lipshitz_diagonal_values = np.array([np.max(np.abs(x)) for x in dataset[f"lip_diag_values_{data_type}"]])
    
    mean_eigen_lipshitz_ratio = eigen_values / lipshitz_diagonal_values
    max_eigen_lipshitz_ratio = eigen_values / max_lipshitz_diagonal_values
    
    return function_values, mean_eigen_lipshitz_ratio, max_eigen_lipshitz_ratio

gd_function_values_list = []
rcd_function_values_list = []
bcd_function_values_list = []

gd_mean_eigen_lipshitz_ratios = []
rcd_mean_eigen_lipshitz_ratios = []
bcd_mean_eigen_lipshitz_ratios = []

gd_max_eigen_lipshitz_ratios = []
rcd_max_eigen_lipshitz_ratios = []
bcd_max_eigen_lipshitz_ratios = []

for experiment in experiments:
    # Process each experiment's data and store the results in respective lists.

    # Create full path to the file
    # It concatenates the given path fragments into a complete path,
    # and automatically uses the correct path separator based on the operating system
    # (e.g., \ on Windows, / on Unix).
    file_path = os.path.join(folder_path, experiment, 'data_dict.pkl')
    
    # Load the data
    data = load(file_path)

    # Process 'gd' and 'rcd' data, separately
    for key in data.keys():
        data[key] = np.array(data[key])  # Convert all data to standard numpy arrays
    
    # Add processed data to their respective lists
    gd_function_values, gd_mean_eigen_lipshitz_ratio, gd_max_eigen_lipshitz_ratio = process_data(data, 'gd')
    rcd_function_values, rcd_mean_eigen_lipshitz_ratio, rcd_max_eigen_lipshitz_ratio = process_data(data, 'rcd')
    bcd_function_values, bcd_mean_eigen_lipshitz_ratio, bcd_max_eigen_lipshitz_ratio = process_data(data, 'bcd')
    
    gd_function_values_list.append(gd_function_values)
    rcd_function_values_list.append(rcd_function_values)
    bcd_function_values_list.append(bcd_function_values)
    
    gd_mean_eigen_lipshitz_ratios.append(gd_mean_eigen_lipshitz_ratio)
    rcd_mean_eigen_lipshitz_ratios.append(rcd_mean_eigen_lipshitz_ratio)
    bcd_mean_eigen_lipshitz_ratios.append(bcd_mean_eigen_lipshitz_ratio)
    
    gd_max_eigen_lipshitz_ratios.append(gd_max_eigen_lipshitz_ratio)
    rcd_max_eigen_lipshitz_ratios.append(rcd_max_eigen_lipshitz_ratio)
    bcd_max_eigen_lipshitz_ratios.append(bcd_max_eigen_lipshitz_ratio)

# Convert the processed data lists to pandas DataFrames for subsequent calculations and plotting.
gd_function_values_df = pd.DataFrame(gd_function_values_list)
rcd_function_values_df = pd.DataFrame(rcd_function_values_list)
bcd_function_values_df = pd.DataFrame(bcd_function_values_list)

gd_mean_ratios_df = pd.DataFrame(gd_mean_eigen_lipshitz_ratios)
rcd_mean_ratios_df = pd.DataFrame(rcd_mean_eigen_lipshitz_ratios)
bcd_mean_ratios_df = pd.DataFrame(bcd_mean_eigen_lipshitz_ratios)

gd_max_ratios_df = pd.DataFrame(gd_max_eigen_lipshitz_ratios)
rcd_max_ratios_df = pd.DataFrame(rcd_max_eigen_lipshitz_ratios)
bcd_max_ratios_df = pd.DataFrame(bcd_max_eigen_lipshitz_ratios)

# Calculate mean and standard deviation for each column (time step)
mean_gd_values = np.array(gd_function_values_df.mean(), dtype=float)
std_gd_values = np.array(gd_function_values_df.std(), dtype=float)
mean_rcd_values = np.array(rcd_function_values_df.mean(), dtype=float)
std_rcd_values = np.array(rcd_function_values_df.std(), dtype=float)
mean_bcd_values = np.array(bcd_function_values_df.mean(), dtype=float)
std_bcd_values = np.array(bcd_function_values_df.std(), dtype=float)

mean_gd_ratios = np.array(gd_mean_ratios_df.mean(), dtype=float)
std_gd_ratios = np.array(gd_mean_ratios_df.std(), dtype=float)
mean_rcd_ratios = np.array(rcd_mean_ratios_df.mean(), dtype=float)
std_rcd_ratios = np.array(rcd_mean_ratios_df.std(), dtype=float)
mean_bcd_ratios = np.array(bcd_mean_ratios_df.mean(), dtype=float)
std_bcd_ratios = np.array(bcd_mean_ratios_df.std(), dtype=float)

mean_gd_max_ratios = np.array(gd_max_ratios_df.mean(), dtype=float)
std_gd_max_ratios = np.array(gd_max_ratios_df.std(), dtype=float)
mean_rcd_max_ratios = np.array(rcd_max_ratios_df.mean(), dtype=float)
std_rcd_max_ratios = np.array(rcd_max_ratios_df.std(), dtype=float)
mean_bcd_max_ratios = np.array(bcd_max_ratios_df.mean(), dtype=float)
std_bcd_max_ratios = np.array(bcd_max_ratios_df.std(), dtype=float)

########################################################
# Plot 1: Change of function values with iterations
# Create new figures and plots
plt.figure()

# Plot mean learning curve for GD, RCD, and BCD function values
plt.plot(mean_gd_values, color='r', label='GD')
plt.plot(mean_rcd_values, color='b', label='RCD')
plt.plot(mean_bcd_values, color='g', label='BCD')

# Plot shaded area for standard deviation of GD, RCD, and BCD function values
plt.fill_between(range(len(mean_gd_values)), 
                 mean_gd_values - std_gd_values, 
                 mean_gd_values + std_gd_values, 
                 color='r', alpha=0.2)
plt.fill_between(range(len(mean_rcd_values)), 
                 mean_rcd_values - std_rcd_values, 
                 mean_rcd_values + std_rcd_values, 
                 color='b', alpha=0.2)
plt.fill_between(range(len(mean_bcd_values)), 
                 mean_bcd_values - std_bcd_values, 
                 mean_bcd_values + std_bcd_values, 
                 color='g', alpha=0.2)

plt.title('Energy Ratio')
plt.xlabel('Iterations')
plt.ylabel('Energy ratio: $E / E_{GS}$')
plt.legend()
if args.x_lim > 0: 
    plt.xlim(0, args.x_lim)
    # The first line checks if args.x_lim is greater than 0,
    # If it is, set the range of the x-axis from 0 to args.x_lim.
plt.grid()  # Add grid lines
plt.tight_layout()  # Optimize the layout of the graph
plt.savefig(f'{plot_path}/energy_HM.png')
# Clear figure after saving
plt.clf()

########################################################
# Plot 2: Change of function values with the number of accumulated function evaluations
# Create new figures and plots
plt.figure()

# Plot mean learning curve for GD, RCD, and BCD function values
# np.arange(len) generates an integer sequence from 0 to len - 1
times_gd = np.arange(len(mean_gd_values)) * args.dim * 2
plt.plot(times_gd, mean_gd_values, color='r', label='GD')
# Since RCD calculates only one partial derivative per iteration; GD calculates args.dim partial derivatives per iteration, so we need to multiply args.dim here

# The x-axis here is the cumulative number of function evaluations!
# The x-axis here is the cumulative number of function evaluations!
# The x-axis here is the cumulative number of function evaluations!

times_rcd = np.arange(len(mean_rcd_values)) * 2
plt.plot(times_rcd, mean_rcd_values, color='b', label='RCD')

times_bcd = np.arange(len(mean_bcd_values)) * 3
plt.plot(times_bcd, mean_bcd_values, color='g', label='BCD')

# Plot shaded area for standard deviation of GD, RCD, and BCD function values
plt.fill_between(times_gd, 
                 mean_gd_values - std_gd_values, 
                 mean_gd_values + std_gd_values, 
                 color='r', alpha=0.2)
plt.fill_between(times_rcd, 
                 mean_rcd_values - std_rcd_values, 
                 mean_rcd_values + std_rcd_values, 
                 color='b', alpha=0.2)
plt.fill_between(times_bcd, 
                 mean_bcd_values - std_bcd_values, 
                 mean_bcd_values + std_bcd_values, 
                 color='g', alpha=0.2)

plt.title('Energy Ratio')
plt.xlabel('Number of function evaluations')
plt.ylabel('Energy ratio: $ E / E_{GS}$')
plt.legend()
if args.x_lim > 0: 
    plt.xlim(0, args.x_lim)
plt.xlim(0, min(max(times_rcd), max(times_bcd), max(times_gd)))
# The second line plt.xlim(0, len(mean_rcd_values)) will always be executed and override the previous setting.
plt.grid()
plt.tight_layout()
plt.savefig(f'{plot_path}/energy_HM_fun_evals.png')
plt.clf()

########################################################
# Plot 3
# Plot mean learning curve for mean eigen Lipschitz ratios
plt.figure()

plt.plot(mean_gd_ratios, color='r', label='GD')
plt.plot(mean_rcd_ratios, color='b', label='RCD')
plt.plot(mean_bcd_ratios, color='g', label='BCD')

plt.fill_between(range(len(mean_gd_ratios)), 
                 mean_gd_ratios - std_gd_ratios, 
                 mean_gd_ratios + std_gd_ratios, 
                 color='r', alpha=0.2)
plt.fill_between(range(len(mean_rcd_ratios)), 
                 mean_rcd_ratios - std_rcd_ratios, 
                 mean_rcd_ratios + std_rcd_ratios, 
                 color='b', alpha=0.2)
plt.fill_between(range(len(mean_bcd_ratios)), 
                 mean_bcd_ratios - std_bcd_ratios, 
                 mean_bcd_ratios + std_bcd_ratios, 
                 color='g', alpha=0.2)

plt.title('Lipschitz Constant Ratio')
plt.xlabel('Steps')
plt.ylabel('L / $L_{avg}$')
if args.x_lim > 0: 
    plt.xlim(0, args.x_lim)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f'{plot_path}/lip_HM.png')
plt.clf()

########################################################
# Plot 4
# Plot mean learning curve for max eigen Lipschitz ratios
plt.figure()

plt.plot(mean_gd_max_ratios, color='r', label='GD')
plt.plot(mean_rcd_max_ratios, color='b', label='RCD')
plt.plot(mean_bcd_max_ratios, color='g', label='BCD')

plt.fill_between(range(len(mean_gd_max_ratios)), 
                 mean_gd_max_ratios - std_gd_max_ratios, 
                 mean_gd_max_ratios + std_gd_max_ratios, 
                 color='r', alpha=0.2)
plt.fill_between(range(len(mean_rcd_max_ratios)), 
                 mean_rcd_max_ratios - std_rcd_max_ratios, 
                 mean_rcd_max_ratios + std_rcd_max_ratios, 
                 color='b', alpha=0.2)
plt.fill_between(range(len(mean_bcd_max_ratios)), 
                 mean_bcd_max_ratios - std_bcd_max_ratios, 
                 mean_bcd_max_ratios + std_bcd_max_ratios, 
                 color='g', alpha=0.2)

plt.title('Lipschitz Constant Ratio')
plt.xlabel('Steps')
plt.ylabel('L / $L_{max}$')
if args.x_lim > 0: 
    plt.xlim(0, args.x_lim)
# plt.ylim(0.0 - 0.01, 1.0 + 0.01)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f'{plot_path}/lip_max_HM.png')
# Clear figure after saving
plt.clf()
