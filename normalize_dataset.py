#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data: Mon May 20 18:12:53 2024
@author: marcalbesa

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import exposure
from PIL import Image

# Function to preprocess and save images
def preprocess_and_save(images_list, conditions_list, output_path):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for i, cond in enumerate(conditions_list):
        cond_path = os.path.join(output_path, cond)
        if not os.path.exists(cond_path):
             os.makedirs(cond_path)

        for j, image in enumerate(images_list[i]):
            # Perform intensity normalization
            normalized_image = exposure.rescale_intensity(image, out_range=(0, 1))
            
            # Convert to uint8 for saving as TIFF
            normalized_image_uint8 = (normalized_image * 255).astype(np.uint8)

            # Create a multipage TIFF
            save_path = os.path.join(cond_path, f"{cond}_{j}.png")
            with Image.fromarray(normalized_image_uint8) as img:
                img.save(save_path)
                print(f"Saved {save_path}")

# Modify your code to call the preprocessing function
if __name__ == "__main__":
    # Images path
    #conditions_list = ['Alive', 'Dead']
    #conditions_list = ['1-0nin_30', '2-0_001nin_30','3-0_01nin_30','4-0_1nin_30','5-1nin_30','6-IC50_30','7-10nin_30','8-100nin_30']
    conditions_list = ['1-Pre24hNT', '2-Post6hNT', '3-Post24hNT', '4-post30hNT', '5-Post48hNT', '6-Post54hNT', '7-Post72hNT']
    files_path = r'C:\EBM\4t CURS\TFG\Datasets\Cells-Lourdes\No tumoral\Outputs\Objects_out'
    
    # List to store the loaded images
    images_list = []
    
    # Directory containing TIFF images
    for cond in conditions_list:
        cond_images = []
        path = os.path.join(files_path, cond)
        if os.path.exists(path):
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                image = plt.imread(file_path)
                image = resize(image, (256, 256, 3))
                cond_images.append(image)
        else:
            print(f"Directory does not exist: {path}")
        images_list.append(cond_images)
    
    # Preprocess and save images
    output_path = r'C:\EBM\4t CURS\TFG\Datasets\Cells-Lourdes\No tumoral\Outputs\Objects_Out_Norm'
    preprocess_and_save(images_list, conditions_list, output_path)
