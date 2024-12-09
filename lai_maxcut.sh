#!/bin/bash

# My computer is too slow, so I can only reduce the number of repeats; otherwise, it takes too long. Default repeat=10
repeat=5
num_iter=500

# Define multiple sigma variable values
# sigma_values=(0.0 0.01 0.02 0.05 0.1 0.2)
# sigma_values=(0.0 0.01)

# Use the seq command to generate a sequence from 0.01 to 0.2 with a step size of 0.01, and store the result in an array
# sigma_values=($(seq 0.00 0.02 0.2))
# sigma_values=(0.0 0.01)
sigma_values=(0.0 0.01 0.05 0.1)
# sigma_values=(0.01)

# Remember, we only use lr_gd to name the folder path!!
for sigma in "${sigma_values[@]}"; do
    python run_maxcut.py --dim 20 --sigma ${sigma} --repeat ${repeat} --lr_gd 0.1 --lr_rcd 0.8 --num_iter ${num_iter}
    python lai_plot_script.py --phys maxcut --lr 0.1 --sigma ${sigma} --dim 20
done

python big_image.py --root_dir "plots/maxcut/lr_0.1/dim_20" --output_dir "plots/maxcut/lr_0.1/dim_20/combined_images"
