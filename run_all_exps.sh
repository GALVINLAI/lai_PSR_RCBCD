#!/bin/bash

# My computer is not powerful, so I have to reduce the number of repeats to save time. Default repeat=10
repeat=10

# Define the sigma variable
sigma=0.05

# Run the python scripts with the defined parameters
# Plot scripts

# Remember, we only use lr_gd to name the folder path!!
python run_tfim.py --lr_gd 0.001 --lr_rcd 0.015 --sigma ${sigma} --dim 20 --num_iter 300 --repeat ${repeat}
python plot_script.py --phys tfim --lr 0.001 --sigma ${sigma} --dim 20

python run_heisenberg.py --lr_gd 0.001 --lr_rcd 0.01 --sigma ${sigma} --dim 28 --num_iter 3000 --repeat ${repeat}
python plot_script.py --phys heisenberg --lr 0.001 --sigma ${sigma} --dim 28

python run_maxcut.py --lr_gd 0.1 --lr_rcd 3.0 --sigma ${sigma} --dim 20 --num_iter 1000 --repeat ${repeat}
python plot_script.py --phys maxcut --lr 0.1 --sigma ${sigma} --dim 20

python run_tsp.py --lr_gd 0.0001 --lr_rcd 0.005 --sigma ${sigma} --dim 90 --num_iter 1000 --repeat ${repeat}
python plot_script.py --phys tsp --lr 0.0001 --sigma ${sigma} --dim 90

python run_factor.py --lr_gd 0.18 --lr_rcd 0.18 --sigma ${sigma} --dim 40 --num_iter 1000 --repeat ${repeat}
python plot_script.py --phys factor --lr 0.18 --sigma ${sigma} --dim 40

# Change sigma and plot again
# sigma=0.14
# python plot_script.py --phys tfim --lr 0.001 --sigma ${sigma} --dim 20
# python plot_script.py --phys heisenberg --lr 0.001 --sigma ${sigma} --dim 28
# python plot_script.py --phys maxcut --lr 0.1 --sigma ${sigma} --dim 20
# python plot_script.py --phys factor --lr 0.18 --sigma ${sigma} --dim 40
# python plot_script.py --phys tsp --lr 0.0001 --sigma ${sigma} --dim 90