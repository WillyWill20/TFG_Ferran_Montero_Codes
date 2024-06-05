# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:13:07 2024

@author: saraf
"""
path = r'C:\EBM\4t CURS\TFG\Datasets\Cells-Lourdes\No tumoral\Outputs\Objects_out'
path_output = path

import os
from PIL import Image

def make_square_and_resize(image_path, output_path, size=512):
    # Open the input image
    img = Image.open(image_path)
    
    # Get the size of the image
    width, height = img.size
    
    # Calculate the size of the new square image
    new_size = max(width, height)
    
    # Create a new black (zero-padded) square image
    new_img = Image.new("RGB", (new_size, new_size), (0, 0, 0))
    
    # Paste the original image onto the center of the new square image
    new_img.paste(img, ((new_size - width) // 2, (new_size - height) // 2))
    
    # Resize the new square image to the desired size (512x512)
    resized_img = new_img.resize((size, size), Image.LANCZOS)
    
    # Save the result as a jpeg
    resized_img.save(output_path, format="PNG")

def process_images(input_folder, output_folder, size=512):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
            make_square_and_resize(image_path, output_path, size)
            print(f"Processed {filename} and saved to {output_path}")

# Example usage
input_folder = path
output_folder = path_output
for folder in os.listdir(path):
    input_path = os.path.join(input_folder,folder)
    output_path = input_path
    process_images(input_path, output_path)

    
    
    
