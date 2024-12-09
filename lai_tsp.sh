#!/bin/bash

# My computer is too slow, so I can only reduce the number of repeats; otherwise, it takes too long. Default repeat=10
repeat=1
num_iter=2000

# Define multiple sigma variable values
# sigma_values=(0.0 0.01 0.02 0.05 0.1 0.2)
# sigma_values=(0.0 0.025 0.05 0.1 0.2)
sigma_values=(0.2)
# sigma_values=(0.1 0.2 0.3 0.4)

# Remember, we only use lr_gd to name the folder path!!
for sigma in "${sigma_values[@]}"; do
    python run_tsp.py --dim 90 --sigma ${sigma} --repeat ${repeat} --lr_gd 0.0001 --lr_rcd 0.001 --num_iter ${num_iter}
    python lai_plot_script.py --phys tsp --lr 0.0001 --sigma ${sigma} --dim 90
done

python big_image.py --root_dir "plots/tsp/lr_0.0001/dim_90" --output_dir "plots/tsp/lr_0.0001/dim_90/combined_images"
