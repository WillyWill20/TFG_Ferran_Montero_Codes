# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(FerranMS)s
"""
#%% Libraries
import os
import numpy as np
import keras
from keras import layers
import tensorflow as tf
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import random

#%% Defining paths
training_data_path = r'C:\EBM\4t CURS\TFG\Datasets\Training_70-30\Train_Normalized'
test_data_path = r'C:\EBM\4t CURS\TFG\Datasets\Training_70-30\Test_Normalized'
weights_path = r'C:\EBM\4t CURS\TFG\Scripts\Checkpoints\SavedModel_70-30'
model_file_path = weights_path

#%% Load the model and it's weights
model = tf.keras.models.load_model(model_file_path)
model.summary()

lr=0.0001

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name='accuracy'),
        AUC(name='auc'),
        Precision(name='precision'),
        Recall(name='recall')])

#%% Functions for predicting correctly
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def preprocess_image(img_path):
    img = load_img(img_path)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_images_from_directory(directory_path, model):
    # target_size = model.input_shape[1:3]  # Assuming input shape is (batch_size, height, width, channels)
    predictions = {}

    # Initialize a dictionary to keep count of '1's in each subfolder
    subfolder_counts = {}

    for root, _, files in os.walk(directory_path):
        # Initialize the count for the current subfolder
        subfolder_counts[root] = 0

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(root, file)
                img_array = preprocess_image(img_path)
                logits = model.predict(img_array)
                probabilities = sigmoid(logits)
                binary_labels = (probabilities > 0.5).astype(int)
                predictions[img_path] = binary_labels[0][0]  # Assuming binary classification, take the first element

                # Increment the count if the prediction is '1'
                if binary_labels[0][0] == 1:
                    subfolder_counts[root] += 1
                    print(f'We found a DEAD cell in c={root}')

    return predictions, subfolder_counts

#%% Predicting images in the specified directory
predictions, subfolder_counts = predict_images_from_directory(test_data_path, model)

#%% Plotting predictions
def plot_random_images(predictions, n_im, label, title, saveIm = False):

    paths = [path for path, pred_label in predictions.items() if pred_label == label]
    
    num_images = n_im
    
    # Select 9 random images
    selected_paths = random.sample(paths, num_images)
    
    plt.figure(figsize=(10, 10))
    for i, img_path in enumerate(selected_paths):
        img = load_img(img_path)
        plt.subplot(int(np.sqrt(num_images)), int(np.sqrt(num_images)), i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(title)
    if saveIm==True:
        plt.savefig(title+'.png')
    plt.show()

#%% Plotting random images with label 0
plot_random_images(predictions, n_im=4, label=0, title='Predicted alive cells', saveIm=True)

#%% Plotting random images with label 1
plot_random_images(predictions, n_im=4, label=1, title='Predicted dead cells', saveIm=True)

#%% Do the IC50 curve
total_cells = []
for C in os.listdir(test_data_path):
    concentrations_folder = os.path.join(test_data_path, C)
    entries = os.listdir(concentrations_folder)
    total_cells.append(len(entries))
    
n_dead_cells = list(subfolder_counts.values())[1:]
n_alive_cells =[]
for i in range(len(total_cells)):
    dead_count = n_dead_cells[i]
    total_count = total_cells[i]
    alive_count = total_count - dead_count
    n_alive_cells.append(alive_count)
    # print(f'Alive: {alive_count}', '\n', f'Dead: {dead_count}', '\n', f'Total: {total_count}', '\n\n')
    
def viability_curve(alive_cells, total_cells, saveIm=False):
    viability_values = []
    for i in range(len(total_cells)):
        viability = alive_cells[i]*100/total_cells[i]
        viability_values.append(viability)
    
    viability_values.pop(5)
    viability_values.pop(0)
    
    X_ax = np.logspace(-3, 2, num=len(viability_values))
    
    plt.figure()
    plt.plot(X_ax,viability_values,color='maroon')
    plt.scatter(X_ax, viability_values, color='maroon', s=50, zorder=5) 
    plt.ylabel('% Viability')
    plt.xlabel('[Nintedanib] (uM)')
    plt.xscale('log')
    plt.grid(True)
    plt.title("Viability curve")
    if saveIm==True:
        plt.savefig('Viability curve 70-30')
    plt.show()
    
    return viability_values

viability_nintedanib = viability_curve(n_alive_cells, total_cells, saveIm=True)  

#%% Obtaining the viability values
deaths_and_concentrations = {}
len_general_path = len(test_data_path)
for path, value in subfolder_counts.items():
    conc = path[len_general_path+3:-3]
    entries = os.listdir(path)
    tot_cells = len(entries)
    alive = tot_cells-value
    deaths_and_concentrations[conc] = alive*100/tot_cells

print(deaths_and_concentrations)

#%% Barplots
plt.figure(figsize = (8, 5))
plt.bar(list(deaths_and_concentrations.keys())[1:], list(deaths_and_concentrations.values())[1:], color='maroon', width = 0.4)
plt.ylabel('% viability',fontsize=12)
plt.title('Viability predicted in cells treated with nintedanib',fontsize=15)
plt.grid(True)
plt.savefig('Viability_predicted_cells_70-30.png')
plt.show()

#%% Save the data of the IC50 curve
Concentrations = [0, 0.001, 0.01, 0.1, 1, 2.915, 10, 100]
#times = list(deaths_and_concentrations.keys())[1:]
Viability = list(deaths_and_concentrations.values())[1:]

dtype = [('Concentrations', float), ('Viability', float)]

data = np.array(list(zip(Concentrations, Viability)), dtype=dtype)

#np.savetxt('times_viability.txt', data, delimiter=';', header='Concentrations;Viability', fmt='%s')

#%%
plt.figure()
plt.plot(Concentrations[1:], Viability[1:],color='maroon')
plt.scatter(Concentrations[1:], Viability[1:],color='maroon')
plt.xscale('log')
plt.ylabel('% Viability')
plt.xlabel('[Nintedanib] (uM)')
plt.title('Viability curve')
plt.grid(True)
plt.savefig('Viability_70-30_all_Concentrations_except_0nin')
plt.show()

#%% Getting the IC50 point:
M = max(Viability[1:])
m = min(Viability[1:])
F = (M-m)*0.5 + m
h = -1
X = [0.001, 0.01, 0.1, 1, 2.915, 10, 100]
log_ic50 = np.array([2.915])
y = m + ((M-m)/(1 + 10**((log_ic50 -X)*h) + np.log10((M-m)/(F-m) +1)))
mid_y = (max(y)-min(y))*0.5 + min(y)

plt.figure(figsize=(8,6))
plt.plot(X,y,'r--', label='Trend Line - Logistic model')
plt.plot(Concentrations[1:], Viability[1:], color='maroon', label='Predicted curve')
plt.scatter(Concentrations[1:], Viability[1:], color='maroon')
plt.axhline(mid_y,color='Black')
#plt.scatter(3.57, mid_y,color='Black')
plt.ylabel('% Viability')
plt.xlabel('[Nintedanib] (uM)')
plt.grid(True)
plt.legend()
plt.xscale('log')
plt.show()

#%% Implementing exponential fit:
from scipy.optimize import curve_fit

def exponential_model(x, a, b):
    return a * np.exp(b * x)

X = np.array([0.001, 0.01, 0.1, 1, 2.915, 10, 100])
y = np.array(Viability[1:])
params, covariance = curve_fit(exponential_model, X, y)
a, b = params
y_fit = exponential_model(X, a, b)

mid_y = (max(y)-min(y))*0.5 + min(y)
IC50 = np.log(mid_y/a)/b

plt.figure(figsize=(6, 4))
#plt.plot(X,y,  label='Predicted curve', color='maroon')
plt.axvline(IC50, color='black', label = 'IC50 calculated')
plt.scatter(X, y, color='maroon')
plt.plot(X, y_fit, 'r-', label='Fitted Exponential Curve')
annotation_text = f'IC50 (uM): {round(IC50, 2)}'
plt.text(x=0.0007,y=83.5,s=annotation_text, fontsize=12, color='Black')
plt.ylabel('% Viability')
plt.xlabel('[Nintedanib] (uM)')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()

print(f'y = {round(a,2)}·exp({round(b,4)}·X)')
print('IC50 (uM): ', round(IC50,2))




























