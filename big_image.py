import os
import glob
import argparse
from PIL import Image, ImageDraw, ImageFont

# This script combines images from multiple directories into a single image for each specified image type.
# It processes directories that contain images categorized by sigma values, generates a grid of images with captions,
# and saves the combined images to a specified output directory.

# Usage example:
# python big_image.py --root_dir "plots/maxcut/lr_0.1/dim_20" --output_dir "plots/maxcut/lr_0.1/dim_20/combined_images"

# Set font path and size
font_path = "arial.ttf"  # Specify the desired font path here
font_size = 30  # Increase font size
font = ImageFont.truetype(font_path, font_size)

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Combine images from multiple directories into a single image.")
parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing sigma directories')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save combined images')

args = parser.parse_args()
root_dir = os.path.relpath(args.root_dir)
output_dir = os.path.relpath(args.output_dir)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all sigma directory paths
sigma_dirs = glob.glob(os.path.join(root_dir, 'sigma_*'))

# List of image filenames
image_filenames = ['energy_HM_fun_evals.png', 'energy_HM.png']

def combine_images(image_filename):
    # Store all target image paths and corresponding sigma directory names
    image_paths = []
    captions = []

    for sigma_dir in sigma_dirs:
        img_path = os.path.join(sigma_dir, image_filename)
        if os.path.exists(img_path):
            image_paths.append(os.path.relpath(img_path))
            captions.append(os.path.basename(sigma_dir))

    if not image_paths:
        print(f"No images found for {image_filename}")
        return

    # Determine the size of each sub-image
    img_sample = Image.open(image_paths[0])
    img_width, img_height = img_sample.size

    # Determine the layout of the combined image
    cols = 5  # Number of images per row
    rows = (len(image_paths) + cols - 1) // cols  # Calculate total rows

    # Create a canvas for the combined image
    canvas_width = cols * img_width
    canvas_height = rows * (img_height + 60)  # Extra 60 pixels for larger titles
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)

    # Paste images onto the canvas and add titles
    for idx, (img_path, caption) in enumerate(zip(image_paths, captions)):
        img = Image.open(img_path)
        x = (idx % cols) * img_width
        y = (idx // cols) * (img_height + 60)
        
        canvas.paste(img, (x, y))
        
        # Add title
        bbox = font.getbbox(caption)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = x + (img_width - text_width) // 2
        text_y = y + img_height + 5  # Add some spacing
        
        draw.text((text_x, text_y), caption, fill="black", font=font)

    # Save the final combined image
    output_path = os.path.join(output_dir, f'combined_{image_filename}')
    canvas.save(output_path)
    print(f"Combined image saved as {output_path}")

# Call combine_images function for each type of image
for image_filename in image_filenames:
    combine_images(image_filename)

