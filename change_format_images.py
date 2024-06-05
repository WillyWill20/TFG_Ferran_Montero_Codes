# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(FerranMS)s
"""

from PIL import Image
import os

def convert_tiff_to_jpg(directory):
    for root, dirs, files in os.walk(directory):
        print(root)
        for file in files:
            if file.endswith('.tiff') or file.endswith('.tif'):
                file_path = os.path.join(root, file)
                img = Image.open(file_path)
                new_file_path = file_path.rsplit('.', 1)[0] + '.png'
                img.convert('RGB').save(new_file_path, 'PNG')
                os.remove(file_path)

dataset_path = r'C:\EBM\4t CURS\TFG\Datasets\Cells-Lourdes\No tumoral\Outputs\Objects_out'
convert_tiff_to_jpg(dataset_path)