# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(FerranMS)s
"""

# =============================================================================
# Data Analysis
# =============================================================================
#%% Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#%% Loading data and preprocessing a little
data_files_path = r'C:\EBM\4t CURS\TFG\Datasets\Images_raw_TFG\Outputs_data'

path0nin_N = '0nin_Nucleis.csv'
path0_001nin_N = '0_001nin_Nucleis.csv'
path0_01nin_N = '0_01nin_Nucleis.csv'
path0_1nin_N = '0_1nin_Nucleis.csv'
path1nin_N = '1nin_Nucleis.csv'
pathIC50_N = 'IC50_Nucleis.csv'
path10nin_N = '10nin_Nucleis.csv'
path100nin_N = '100nin_Nucleis.csv'
# pathpH4_40min_N = 'pH4_40min_Nucleis.csv'
# pathpH4_20min_N = 'pH4_20min_Nucleis.csv'
# pathpH1_20min_N = 'pH1_20min_Nucleis.csv'

path0nin_O = '0nin_Nucleis.csv'
path0_001nin_O = '0_001nin_Objects_R.csv'
path0_01nin_O = '0_01nin_Objects_R.csv'
path0_1nin_O = '0_1nin_Objects_R.csv'
path1nin_O = '1nin_Objects_R.csv'
pathIC50_O = 'IC50_Objects_R.csv'
path10nin_O = '10nin_Objects_R.csv'
path100nin_O = '100nin_Objects_R.csv'
# pathpH4_40min_O = 'pH4_40min_Objects_R.csv'
# pathpH4_20min_O = 'pH4_20min_Objects_R.csv'
# pathpH1_20min_O = 'pH1_20min_Objects_R.csv'

paths = list([path0nin_N,
             path0nin_O,
             path0_001nin_N,
             path0_001nin_O,
             path0_01nin_N,
             path0_01nin_O,
             path0_1nin_N,
             path0_1nin_O,
             path1nin_N,
             path1nin_O,
             pathIC50_N,
             pathIC50_O,
             path10nin_N,
             path10nin_O,
             path100nin_N,
             path100nin_O,
             # pathpH4_40min_N,
             # pathpH4_40min_O,
             # pathpH4_20min_N,
             # pathpH4_20min_O
             ])

data_frames = []

for p in paths:
    path = os.path.join(data_files_path,p)
    df = pd.read_csv(path)
    df_noNaNs = df.dropna(axis=1,how='any')
    data_frames.append(df_noNaNs)
    
nucs_0nin = data_frames[0]
objs_0nin = data_frames[1]
nucs_0_001nin = data_frames[2]
objs_0_001nin = data_frames[3]
nucs_0_01nin = data_frames[4]
objs_0_01nin = data_frames[5]
nucs_0_1nin = data_frames[6]
objs_0_1nin = data_frames[7]
nucs_1nin = data_frames[8]
objs_1nin = data_frames[9]
nucs_IC50 = data_frames[10]
objs_IC50 = data_frames[11]
nucs_10nin = data_frames[12]
objs_10nin = data_frames[13]
nucs_100nin = data_frames[14]
objs_100nin = data_frames[15]
# nucs_ph4_40 = data_frames[16]
# objs_ph4_40 = data_frames[17]
# nucs_ph4_20 = data_frames[18]
# objs_ph4_20 = data_frames[19]

dfs_of_nucleis = [nucs_0nin,nucs_0_001nin,nucs_0_01nin,nucs_0_1nin,nucs_1nin,nucs_IC50,nucs_10nin,nucs_100nin,#nucs_ph4_40,nucs_ph4_20
                  ]
dfs_of_objs = [objs_0nin,objs_0_001nin,objs_0_01nin,objs_0_1nin,objs_1nin,objs_IC50,objs_10nin,objs_100nin,#objs_ph4_40,objs_ph4_20
               ]

for df in dfs_of_nucleis:
    df.columns = [col.replace('AreaShape_', '') for col in df.columns]

for df in dfs_of_objs:
    df.columns = [col.replace('AreaShape_', '') for col in df.columns]
    
concentration_mapping = [0, 0.001, 0.01, 0.1, 1, 2.915, 10, 100, 440, 420]
for i in range(len(dfs_of_objs)):
    concentration = concentration_mapping[i]
    dfs_of_nucleis[i]['Concentration'] = concentration
    dfs_of_objs[i]['Concentration'] = concentration

nucs_df = pd.concat(dfs_of_nucleis)
objs_df = nucs_df.dropna(axis=1,how='any')
columns_with_strings = nucs_df.select_dtypes(include='object').columns
nucs_df = nucs_df.drop(columns=columns_with_strings)

objs_df = pd.concat(dfs_of_objs)
objs_df = objs_df.dropna(axis=1,how='any')
columns_with_strings = objs_df.select_dtypes(include='object').columns
objs_df = objs_df.drop(columns=columns_with_strings)

#%% 6 plots for each parameter in Objects
objs_col_names = objs_df.columns

for col in objs_col_names:
    sorted_param_O = sorted(list(objs_df[col]))
    
    plt.figure(figsize=(8,4))
    
    # plt.subplot(2,3,1)
    # plt.plot(objs_df[col],c='Red')
    # #plt.hlines(np.median(objs_df[col]),color='darkgreen',xmin=0,xmax=len(objs_df[col]))
    # plt.grid(True)
    
    # plt.subplot(2,3,2)
    # plt.scatter(range(len(sorted_param_O)),sorted_param_O, c='Red',s=3)
    # plt.hlines(np.median(objs_df[col]),color='darkgreen',xmin=0,xmax=len(objs_df[col]))
    # plt.xlabel('{}'.format(col))
    # plt.grid(True)
    
    # plt.subplot(2,3,3)
    # plt.hist(objs_df[col],bins=50, color='Red', edgecolor='black')
    # plt.axvline(x=np.median(objs_df[col]), color='darkgreen')
    # plt.grid(True)
    
    plt.subplot(1,3,1)
    plt.plot(np.log10(objs_df[col]+1e-6),c='Red')
    #plt.hlines(np.median(np.log(objs_df[col])),color='darkgreen',xmin=0,xmax=len(objs_df[col]))
    plt.xlabel('Cells')
    plt.grid(True)
    
    plt.subplot(1,3,2)
    plt.scatter(range(len(sorted_param_O)),np.log10(sorted_param_O), c='Red', s=3)
    plt.hlines(np.median(np.log10(objs_df[col])),color='darkgreen',xmin=0,xmax=len(objs_df[col]))
    plt.xlabel('Cell index')
    plt.grid(True)
    
    plt.subplot(1,3,3)
    plt.hist(np.log10(objs_df[col]+1e-6),bins=50, color='Red', edgecolor='black')
    plt.axvline(x=np.median(np.log10(objs_df[col])), color='darkgreen')
    plt.xlabel('Values')
    plt.ylabel('Counts')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

#%% 6 plots for each parameter in Nucleus
nucs_col_names = nucs_df.columns

for col in nucs_col_names:
    sorted_param_N = sorted(list(nucs_df[col]))
    
    plt.figure(figsize=(8,4))
    
    # plt.subplot(2,3,1)
    # plt.plot(nucs_df[col], c='Blue')
    # #plt.hlines(np.median(nucs_df[col]), color='Green', xmin=0, xmax=len(nucs_df[col]))
    # plt.grid(True)
    
    # plt.subplot(2,3,2)
    # plt.scatter(range(len(sorted_param_N)),sorted_param_N, c='Blue',s=3)
    # plt.hlines(np.median(nucs_df[col]), color='magenta', xmin=0, xmax=len(nucs_df[col]))
    # plt.xlabel('{}'.format(col))
    # plt.grid(True)
    
    # plt.subplot(2,3,3)
    # plt.hist(nucs_df[col], bins=50, color='Blue', edgecolor='black')
    # plt.axvline(x=np.median(nucs_df[col]), color='magenta')
    # plt.grid(True)
    
    plt.subplot(1,3,1)
    plt.plot(np.log10(nucs_df[col]+1e-6), c='Blue')
    #plt.hlines(np.median(np.log(nucs_df[col])), color='Green', xmin=0, xmax=len(nucs_df[col]))
    plt.xlabel('Nucleus')
    plt.grid(True)
    
    plt.subplot(1,3,2)
    plt.scatter(range(len(sorted_param_N)),np.log10(sorted_param_N), c='Blue', s=3)
    plt.hlines(np.median(np.log10(nucs_df[col])), color='magenta', xmin=0, xmax=len(nucs_df[col]))
    plt.xlabel('Nuclei index')
    plt.grid(True)
    
    plt.subplot(1,3,3)
    plt.hist(np.log10(nucs_df[col]+1e-6), bins=50, color='Blue', edgecolor='black')
    plt.axvline(x=np.median(np.log10(nucs_df[col])), color='magenta')
    plt.xlabel('Values')
    plt.ylabel('Counts')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
#%% Exploring boxplots to see the dispersion of each variable
for col in objs_col_names:
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plotting the boxplot for objs_df
    axs[0, 0].boxplot(objs_df[col], notch=True)
    axs[0, 0].set_title('Cells')
    axs[0, 0].set_xlabel(col)
    axs[0, 0].grid(True)
    
    # Plotting the boxplot for nucs_df
    axs[0, 1].boxplot(nucs_df[col], notch=True)
    axs[0, 1].set_title('Nucleus')
    axs[0, 1].set_xlabel(col)
    axs[0, 1].grid(True)
    
    # Log10 boxplots for objs_df
    axs[1, 0].boxplot(np.log10(objs_df[col]), notch=True)
    axs[1, 0].set_title('Log10 Cells')
    axs[1, 0].set_xlabel(col)
    axs[1, 0].grid(True)
    
    # Log10 boxplots for nucs_df
    axs[1, 1].boxplot(np.log10(nucs_df[col]), notch=True)
    axs[1, 1].set_title('Log10 Nucleus')
    axs[1, 1].set_xlabel(col)
    axs[1, 1].grid(True)
    
    # Overall title for the two plots
    plt.suptitle('{}'.format(col))
    plt.show()

#%% Sorted scatter plots colored by concecntration on Objects
for col in objs_df.columns:
    fig, axs = plt.subplots(1, 1, figsize=(7, 5))

    # # Plot the non-logarithmic subplot
    # for conc in objs_df['Concentration'].unique():
    #     conc_df = objs_df[objs_df['Concentration'] == conc]
    #     axs[0].scatter(range(len(conc_df[col])), sorted(list(conc_df[col])), 
    #                    label=f'{conc}')

    # axs[0].set_title(f'{col} - Linear')
    # axs[0].set_xlabel('Index')
    # axs[0].set_ylabel(col)

    # # Plot the logarithmic subplot
    # axs.set_title(f'{col}')
    axs.set_xlabel('Index')
    axs.set_ylabel(col)
    
    for conc in objs_df['Concentration'].unique():
        conc_df = objs_df[objs_df['Concentration'] == conc]
        axs.scatter(range(len(conc_df[col])), sorted(list(np.log(conc_df[col]))), label=f'{conc}')

    axs.legend()

    plt.tight_layout()
    plt.show()
    
#%% Sorted scatter plots colored by concecntration on Nucleis
for col in nucs_df.columns:
    fig, axs = plt.subplots(1, 1, figsize=(7,5))

    # # Plot the non-logarithmic subplot
    # for conc in nucs_df['Concentration'].unique():
    #     conc_df = nucs_df[nucs_df['Concentration'] == conc]
    #     axs[0].scatter(range(len(conc_df[col])), sorted(list(conc_df[col])), 
    #                    label=f'{conc}')

    # axs[0].set_title(f'{col} - Linear')
    # axs[0].set_xlabel('Index')
    # axs[0].set_ylabel(col)

    # # Plot the logarithmic subplot
    # axs[1].set_title(f'{col} - Logarithmic')
    axs.set_xlabel('Index')
    axs.set_ylabel(col)
    
    for conc in nucs_df['Concentration'].unique():
        conc_df = nucs_df[nucs_df['Concentration'] == conc]
        axs.scatter(range(len(conc_df[col])), sorted(list(np.log(conc_df[col]))), label=f'{conc}')

    axs.legend()

    plt.tight_layout()
    plt.show()

#%% Time for labeling manually each row since there is no way to recognise dead from alive just looking at data jajaja
# We will put three labels, which will also help clean the data:
    
    # 3 = Not a cell
    # 2 = More than one cell
    # 1 = Alive
    # 0 = Dead

# What we will do is to pick 50 instances per class (200 in total), and we will
# set the corresponding label in a new column called 'Label', and the ones that
# are not in the manually labeled class we will give them a value of 4 that 
# we will remove later and would serve the pourpose of taking the training set 
# for the random forest algorithm.

Label_column = 4*np.ones(len(objs_df))

labels_0nin = [3,3,1,1,1,3,1,1,2,1,1,3,3,1,3,3,3,3,1,1,1,3,1,1,3,3,1,3,1,2,1,3,1,1,1,1,1,1,3,1,1,1,3,1,1,1,1,1,1,3,1,3,3,1,1,2,1,1,1,3,1,3,3,1,2,3,3,3,3,1,1,3,3,3,1,2,3,2,1,2,1,1,3,3,3,1,1,1,3,3,3,1,1,3,1,1,1,3,1,3,3,3,1,1,1,3,2,1,3,1,3,2,1,1,3,1,1,1,3,3,1,1,1,2,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,1,3,1,1,1,1,3,1,1,1,2,1,3,0,1,1,1,1,1,1,1,3,1,3,1,1,1,1,3,1,1,3,1,1,1,1,1,2,0,3,1,1,0,1,2,1,1,1,1,3,0,1,3,1,3,3,3,1,1,3,2,1,1,2,2,3,3,3,1,1,3,1,3,1,1,1,1,2,1,3,2,1,3,1,3,3,3,1,0,3,3,3,3,3,3,3,3,1,2,1,3,2,1,3,1,0,1,2,1,1,1,1,1,1,1,1,2,1,1,3,2,1,3,1,3,1,3,1,1,3,1,1,1,1,3,1,1,2,1,1,1,1,1,1,3,3,2,2,3,3,1,3,1,3,1,3,3,3,1,1,2,2,1,1,3,0,1,1,1,1,1,3,1,1,1,3,1,1,1,3,1,3,3,3,1,1,1,1,3,3,3,1,1,1,3,3,1,1,3,1,3,1,3,3,1,3,1]

labels_100nin = [1,3,1,1,1,1,1,2,3,0,1,1,2,1,3,0,1,1,1,1,1,3,2,1,1,1,1,3,1,1,1,1,1,1,1,3,1,1,1,1,1,1,2,2,1,3,2,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,3,1,0,1,1,1,2,1,1,1,1,2,2,1,2,1,2,1,2,1,2,1,1,3,2,0,3,0,2,1,2,2,1,1,1,2,1,1,1,1,1,1,1,1,1,3,0,1,2,2,2,1,2,1,1,2,2,2,1,1,2,1,2,2,3,1,1,1,1,2,2,1,2,2,3,3,1,2,2,1,1,2,2,2,1,1,1,1,1,2,3,3,1,3,2,2,2,3,1,3,3,2,2,1,1,2,3,1,2,2,1,1,3,3,3,3,3,2,3,3,3,2,3,3,3,3,3,1,0,2,2,1,2,1,2,2,1,2,1,2,2,2,1,1,3,2,2,2,3,1,2,1,2,2,1,2,1,2,1,1,2,0,1,2,2,1,2,3,2,2,3,3,1,1,1,3,1,2,3,2,1,1,1,1,2,1,1,1,3,3,3,3]

#%% Appending the column of labels to the Objs_df
Label_column[0:len(labels_0nin)] = labels_0nin
Label_column[len(objs_df)-len(objs_ph4_20)-len(objs_ph4_40)-len(objs_100nin):len(objs_df)-len(objs_ph4_20)-len(objs_ph4_40)] = labels_100nin
Label_column[len(objs_0nin)+386] = 0
Label_column[len(objs_0nin)+len(objs_0_001nin)+236] = 0
Label_column[len(objs_0nin)+len(objs_0_001nin)+344] = 0
Label_column[len(objs_0nin)+len(objs_0_001nin)+len(objs_0_01nin)+len(objs_0_1nin)+len(objs_1nin)+141] = 0
Label_column[len(objs_0nin)+len(objs_0_001nin)+len(objs_0_01nin)+len(objs_0_1nin)+len(objs_1nin)+143] = 0
Label_column[len(objs_0nin)+len(objs_0_001nin)+len(objs_0_01nin)+len(objs_0_1nin)+len(objs_1nin)+182] = 0
Label_column[len(objs_0nin)+len(objs_0_001nin)+len(objs_0_01nin)+len(objs_0_1nin)+len(objs_1nin)+250] = 0
Label_column[len(objs_0nin)+len(objs_0_001nin)+len(objs_0_01nin)+len(objs_0_1nin)+len(objs_1nin)+265] = 0
Label_column[len(objs_0nin)+len(objs_0_001nin)+len(objs_0_01nin)+len(objs_0_1nin)+len(objs_1nin)+317] = 0
Label_column[len(objs_0nin)+len(objs_0_001nin)+len(objs_0_01nin)+len(objs_0_1nin)+len(objs_1nin)+len(objs_IC50)+len(objs_10nin)+len(objs_100nin)+5] = 0
Label_column[len(objs_0nin)+len(objs_0_001nin)+len(objs_0_01nin)+len(objs_0_1nin)+len(objs_1nin)+len(objs_IC50)+len(objs_10nin)+len(objs_100nin)+73] = 0
Label_column[len(objs_0nin)+len(objs_0_001nin)+len(objs_0_01nin)+len(objs_0_1nin)+len(objs_1nin)+len(objs_IC50)+len(objs_10nin)+len(objs_100nin)+141] = 0
Label_column[len(objs_0nin)+len(objs_0_001nin)+len(objs_0_01nin)+len(objs_0_1nin)+len(objs_1nin)+len(objs_IC50)+len(objs_10nin)+len(objs_100nin)+148] = 0
Label_column[len(objs_0nin)+len(objs_0_001nin)+len(objs_0_01nin)+len(objs_0_1nin)+len(objs_1nin)+len(objs_IC50)+len(objs_10nin)+len(objs_100nin)+158] = 0
Label_column[len(objs_0nin)+len(objs_0_001nin)+len(objs_0_01nin)+len(objs_0_1nin)+len(objs_1nin)+len(objs_IC50)+len(objs_10nin)+len(objs_100nin)+159] = 0
Label_column[len(objs_0nin)+len(objs_0_001nin)+len(objs_0_01nin)+len(objs_0_1nin)+len(objs_1nin)+len(objs_IC50)+len(objs_10nin)+len(objs_100nin)+165] = 0

objs_df['Label'] = Label_column

#%% Getting the data for the training of the algorithm

data_algorithm = pd.DataFrame.copy(objs_df)

plt.figure()
plt.hist(data_algorithm['Label'],bins=5,edgecolor='black')
plt.grid()
plt.show()

labeled_data = data_algorithm[data_algorithm['Label'].isin([0, 1, 2, 3])]

array_labeled_data = np.array(labeled_data)
X = array_labeled_data[:,2:-4] # we excluded some parameters that had no relevance for the matter
Y = array_labeled_data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.05)

# Rearrange data into DataFrames suitable for the classifier
col_names_clf = objs_col_names[2:-3]
X_train = pd.DataFrame(X_train,columns=col_names_clf)
y_train = pd.DataFrame(y_train,columns = ['Label'])
y_train = y_train.values.ravel()
X_test = pd.DataFrame(X_test,columns = col_names_clf)
y_test = pd.DataFrame(y_test,columns = ['Label'])

plt.figure(figsize=(5,5))
plt.hist(y_train,label='train',color='maroon')
plt.hist(y_test,label='test',color='Red')
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.legend()
plt.show()

#%% Training the RFC model

clf_RFC = RandomForestClassifier(n_estimators = 200, verbose = 1)
clf_RFC.fit(X_train, y_train)

# Predict labels for the test set
y_pred = clf_RFC.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0,1,2,3])

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

labels_conf_mat = ['Dead', 'Alive', '+1 cell', 'Not a cell']
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels_conf_mat)
disp.plot(cmap='Reds', values_format='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
print(conf_matrix)

#%% Labeling the rest of the data:
unlabeled_data = data_algorithm[data_algorithm['Label'].isin([4])]
array_unlabeled_data = np.array(unlabeled_data)
X_unlabeled = array_unlabeled_data[:,2:-4]
col_names_clf = objs_col_names[2:-3]
X_unlabeled = pd.DataFrame(X_unlabeled,columns=col_names_clf)
labels_unlabeled_data = clf_RFC.predict(X_unlabeled)


array_objs_df = np.array(objs_df)
idx = 0

for i in range(len(array_objs_df)): # we are iterating over all the cells with the unlabeled (4) label
    label = array_objs_df[i,-1]
    if label == 4:
        new_label = labels_unlabeled_data[idx]
        array_objs_df[i,-1] = new_label
        idx = idx + 1

#%%
objs_col_names = objs_col_names.append(pd.Index(['Label']))
final_objs_df = pd.DataFrame(array_objs_df, columns = objs_col_names)
#%% Save the dataframe obtained
final_objs_df.to_excel('Final_df.xlsx', index=False)














