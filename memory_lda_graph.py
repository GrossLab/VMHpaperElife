# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:45:25 2019

This script is used in order to apply and visualize the reduction dimensionality method (LDA) on memory files.
It opens and reads a csv file saved in a folder.
For each recorded mouse we created a different folder that contains the all the memory recordings
(two days and for each of them behaviours and position). Here we chose to analyse the different positions of
the experimental mouse during the recordings (home, corrifar, corrihome, farchamber). Moreover we
changed the positions in order to obtain three categories (home, corridor, farchamber).
The aim of this analysis is to understand if exist clusters of neuronal activity in different locations of the cage.

@author: penna
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def reading(file_in, sepfile):

    """
    Function that opens and reads a csv file, calculates z-scored values, and creates a list of three categories that
    correspond to the position of the mouse during the experiment.

    :param file_in: file path of csv file containing memory recordings
    :param sepfile: delimiter of csv file
    :return: a tupla that contains a dataframe with z-scored values and a list of strings containing the position of the
            mouse during the experiment
    """

    data_frame = pd.read_csv(file_in, sep=sepfile)
    data_frame = data_frame.drop('Frames', 1)
    location_column = data_frame.loc[:, 'Beh']
    data_frame = data_frame.drop('Beh', 1)
    neurons_list = []
    for neuron in data_frame:
        neurons_list.append(neuron)
    data_frame = pd.DataFrame(StandardScaler().fit_transform(data_frame), columns=neurons_list)
    for i in range(len(location_column)):
        location = location_column[i]
        if location == 'corriHome' or location == 'CorriFar':
            location_column[i] = 'corridor'
    return data_frame, location_column


def graphing(dataframe_after_LDA, title, color_list, labels):

    """
    Function that given a dataframe after the LDA, returns a graph where each point corresponds to a frame, and each
    cluster correspond to a different location of the mouse during the experiment (HOME, CORRIDOR, FARCHAMBER)

    :param dataframe_after_LDA: dataframe after the dimensionality reduction method
    :param title: image title
    :param color_list: list containing the color of each locations
    :param labels: column of rows that refer to the location of the mouse during the experiment
    :return:
    """

    dic = {'H': 'cornflowerblue', 'C': 'gray', 'FC': 'lightcoral'}
    j = 1
    p = 0.09
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('LDA 1', fontsize=10)
    ax.set_ylabel('LDA 2', fontsize=10)
    ax.set_xlim([-9, 9])
    ax.set_ylim([-5, 7])
    ax.set_title(title, fontsize=15)
    targets = ['Home', 'corridor', 'Farchamber']
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for target, color in zip(targets, color_list):
        if target == 'Home':
            indices_to_keep = dataframe_after_LDA[labels] == target
            ax.scatter(dataframe_after_LDA.loc[indices_to_keep, 0]
                       , dataframe_after_LDA.loc[indices_to_keep, 1]
                       , c='cornflowerblue'
                       , s=10
                       , alpha=0.8)
        elif target == 'corridor':
            indices_to_keep = dataframe_after_LDA[labels] == target
            ax.scatter(dataframe_after_LDA.loc[indices_to_keep, 0]
                       , dataframe_after_LDA.loc[indices_to_keep, 1]
                       , c='gray'
                       , s=10
                       , alpha=0.8)
        else:
            indices_to_keep = dataframe_after_LDA[labels] == target
            ax.scatter(dataframe_after_LDA.loc[indices_to_keep, 0]
                       , dataframe_after_LDA.loc[indices_to_keep, 1]
                       , c='lightcoral'
                       , s=10
                       , alpha=0.8)
    for beh in sorted(dic.keys()):
        legend_patch = patches.Patch(color=dic.get(beh), label=beh)
        legend = plt.legend(handles=[legend_patch], loc=1, bbox_to_anchor=(0.13, p * 8), prop={'size': 10})
        p += 0.01
        if j != len(dic.keys()):
            j += 1
            ax = plt.gca().add_artist(legend)
        else:
            return fig


'''From here we call the function described above: after the reading function, we apply the Linear Discriminant Analysis
 on our dataset, and after that we call graphing function that gives the graph of LDA'''

path = r'D:\Lavoro\roba per piotr\memory files\memory 23\Vmh23M20loc.csv'
data, position = reading(path, ';')
sklearn_lda = LDA(n_components=2)
X_lda = sklearn_lda.fit_transform(data, position)
df = pd.DataFrame(X_lda)
df['target'] = position
colors = ['cornflowerblue','gray','lightcoral']
im = graphing(df, 'Habituation', colors, 'target')
plt.show(im)

