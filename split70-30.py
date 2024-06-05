# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(FerranMS)s
"""
#%% Libraries
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import shutil
import math

#%%
def split_and_copy_images(src_folder, dest_folder_70, dest_folder_30, percentage):
    if not os.path.exists(dest_folder_70):
        os.makedirs(dest_folder_70)
    if not os.path.exists(dest_folder_30):
        os.makedirs(dest_folder_30)
    
    # List all files in the source folder
    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    
    # Calculate the number of files for the 70% split
    num_files_70 = math.floor(len(all_files) * percentage)
    
    # Randomly shuffle files
    random.shuffle(all_files)
    
    # Split the files
    files_70 = all_files[:num_files_70]
    files_30 = all_files[num_files_70:]
    
    # Copy files to the 70% destination folder
    for file_name in files_70:
        src_file = os.path.join(src_folder, file_name)
        dest_file = os.path.join(dest_folder_70, file_name)
        shutil.copy2(src_file, dest_file)
    
    # Copy files to the 30% destination folder
    for file_name in files_30:
        src_file = os.path.join(src_folder, file_name)
        dest_file = os.path.join(dest_folder_30, file_name)
        shutil.copy2(src_file, dest_file)

def process_directory(root_directory, percentage=0.7):
    # Traverse the directory
    for root, dirs, files in os.walk(root_directory):
        for dir_name in dirs:
            src_folder = os.path.join(root, dir_name)
            dest_folder_70 = src_folder + '_70'
            dest_folder_30 = src_folder + '_30'
            split_and_copy_images(src_folder, dest_folder_70, dest_folder_30, percentage)
            
#%%
root_directory = r'C:\EBM\4t CURS\TFG\Datasets\Training_70-30'
process_directory(root_directory)

















