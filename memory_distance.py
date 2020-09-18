# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:45:25 2019

This script finds the distance between clusters of position obtained with the Linear Discriminant Analysis.
It is applied on csv files that contain the recordings of memory days, and returns an heatmap saved in a folder
that has to be specified in the absolute path.
The file csv has to follow a specific name convention 'Vmh'+ 'number that identifies the mouse'+ 'a letter (H or M)' +
'2 or 1' + '0' + 'a string (dist or loc)' (for example: Vmh19H10loc ).

@author: penna
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



def reading(file_in, sepfile):

    """
    Function that open and read csv file, calculates z-scored values, and creates a list of three categories that correspond
    to the position of the mouse during the experiment.

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


def find_distance(df1, df2):

    """
    Function that finds the averaged distance between the elements of two dataframe.

    :param df1: first dataframe
    :param df2: second dataframe
    :return: averaged distance between df1 and df2
    """

    d = 0
    count = 0
    distances = []
    for i in range(len(df1)):
        for j in range(len(df2)):
            try:
                d = d + distance.euclidean((df1[0][i], df1[1][i]), (df2[0][j], df2[1][j]))
            except:
                print("i=   " + str(i) + "   j=   " + str(j) + "  An error occurred...\n exiting the program...")
                exit(1)
            count = count + 1
        d2 = d / count
        distances.append(d2)
        d = 0
        count = 0
    dist = sum(distances) / len(distances)
    return dist


def create_df_for_hm(list_dist, list_column_name):

    """
    Function that creates a dataframe containing the distance between each cluster of position; this dataframe is then
    used to create the heatmap that allows to visualize the distance between each cluster

    :param list_dist: list that contains the distances between each cluster of position (home vs corridor,
            home vs farchamber, farchamber vs corridor)
    :param list_column_name: columns name for the dataframe (H, C, FC)
    :return: a df that in each cells contains the distance between each cluster of position, and has the same columns and
             the same indexes
"""

    df = pd.DataFrame(index=['H', 'C', 'FC'])
    for i in range(len(list_dist)):
        df[list_column_name[i]] = list_dist[i]
    return df


def heatmap_of_distance_creation(dataframe_after_lda, mouse_id):

    """
    Function that creates for each mouse recorded an heatmap that allows to visualize the distance between each cluster of
    position. This function saves the hm into a folder specified by an absolute path

    :param dataframe_after_lda: dataframe after the dimensionality reduction method(LDA)
    :param mouse_id: number of mouse
    :return: a list of list containing the distances between each cluster of position
    """

    df_home = dataframe_after_lda.loc[dataframe_after_lda['state'] == 'Home']
    df_home.index = range(0, len(df_home))
    df_corridor = dataframe_after_lda.loc[dataframe_after_lda['state'] == 'corridor']
    df_corridor.index = range(0, len(df_corridor))
    df_farchamber = dataframe_after_lda.loc[dataframe_after_lda['state'] == 'Farchamber']
    df_farchamber.index = range(0, len(df_farchamber))
    dist_home_corridor = find_distance(df_home, df_corridor)
    dist_home_farchamber = find_distance(df_home, df_farchamber)
    home_distance = (0, dist_home_corridor, dist_home_farchamber,)
    dist_corridor_farchamber = find_distance(df_corridor, df_farchamber)
    corridor_distance = (dist_home_corridor, 0, dist_corridor_farchamber)
    farchamber_distance = (dist_home_farchamber, dist_corridor_farchamber, 0)
    list_of_dist = [home_distance, corridor_distance, farchamber_distance]
    list_of_labels = ['H', 'C', 'FC']
    df_hm = create_df_for_hm(list_of_dist, list_of_labels)
    plt.subplot(231)
    ax = plt.axes()
    cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
    mask = np.zeros_like(df_hm, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    np.fill_diagonal(mask, 0)
    sns.set(style='white', font_scale=2)
    sm = sns.heatmap(df_hm, annot=True, vmin=2, vmax=10, mask=mask, cmap=cmap, annot_kws={'size': 24})
    ax.set_title('Mouse' + mouse_id + ' Distance HeatMap')
    figure = sm.get_figure()
    figure.savefig(r'C:\Users\penna\Desktop\others' + '\Mouse' + mouse_id + 'DistanceHeatMap', dpi=400)
    return list_of_dist


'''From here we call the functions described above: after reading we apply the Linear Discriminant Analysis on each csv
file; then we find the id_number of each mouse and we call heatmap_of_distance_creation, that, after the distance 
calculation, saves the hm for each mouse'''

distance_list = []
path = r'D:\Lavoro\roba per piotr\memory files\Memory everything'
for file in os.listdir(path):
    data, locations = reading(path + '\\' + file, ';')
    sklearn_lda = LDA(n_components=2)
    X_lda = sklearn_lda.fit_transform(data, locations)
    df = pd.DataFrame(X_lda)
    df['state'] = locations
    mouse_num = file[3:5]
    try:
        int(mouse_num)
    except:
        mouse_num = mouse_num[0]
    distance_list.append(heatmap_of_distance_creation(df, mouse_num))
