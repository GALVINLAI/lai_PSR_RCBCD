#!/bin/bash

# My computer is too slow, so I can only reduce the number of repeats; otherwise, it takes too long. Default repeat=10
repeat=1
num_iter=1000

# Define multiple sigma variable values
# sigma_values=(0.0 0.01 0.02 0.05 0.1 0.2)
# sigma_values=(0.0 0.025 0.05 0.1 0.2)
sigma_values=(0.05)
# sigma_values=(0.1 0.2 0.3 0.4)

# Remember, we only use lr_gd to name the folder path!!
for sigma in "${sigma_values[@]}"; do
    python run_factor.py --dim 40 --sigma ${sigma} --repeat ${repeat} --lr_gd 0.18 --lr_rcd 0.18 --num_iter ${num_iter}
    python lai_plot_script.py --phys factor --lr 0.18 --sigma ${sigma} --dim 40
done

python big_image.py --root_dir "plots/tsp/lr_0.18/dim_40" --output_dir "plots/tsp/lr_0.18/dim_40/combined_images"
