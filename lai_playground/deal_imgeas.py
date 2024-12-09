import os
import shutil
import re

# This script organizes images from a source directory into subdirectories based on their filenames.
# The filenames follow the pattern iter_#1_cor_#2.png, where #2 is used to determine the subdirectory name.

# Set the source directory containing the images
source_directory = "bcd_within_ding_plots_heisenberg"  # Modify this to your image directory
# Set the target directory where the organized images will be saved
target_directory = "bcd_within_ding_plots_heisenberg_sorted"  # Modify this to your desired save directory

# Regular expression pattern to match filenames of the format iter_#1_cor_#2.png
pattern = re.compile(r"iter_(\d+)_cor_(\d+)\.png")

# Check if the target directory exists, if not, create it
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Iterate over all files in the source directory
for filename in os.listdir(source_directory):
    match = pattern.match(filename)
    if match:
        # Extract the value of #2
        cor_value = match.group(2)
        
        # Create a folder named after the value of #2
        cor_folder = os.path.join(target_directory, f"cor_{cor_value}")
        if not os.path.exists(cor_folder):
            os.makedirs(cor_folder)
        
        # Move the file to the corresponding folder
        source_path = os.path.join(source_directory, filename)
        target_path = os.path.join(cor_folder, filename)
        shutil.move(source_path, target_path)
        print(f"Moved {filename} to {cor_folder}")

print("Images have been successfully classified.")
