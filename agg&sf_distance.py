# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 09:26:43 2018

This script finds the distance between clusters of experimental days obtained with the Linear Discriminant Analysis.
It is applied on csv files that contain the recordings of aggression and social fear days, and returns heatmaps saved in
a folder that has to be specified in an absolute path.
Moreover it finds the averaged distance obtained analysing all mice given in input.
The csv file has to follow a specific name convention: 'Vmh' + 'id_numb of mouse' + 'a string (A or SF )' + 'a numb that
identifies the day of recording' + 'a number that identifies the number of recording' + 'a string (beh)' 23A10beh
Moreover the folders that contains the file of each mouse has to be named in the following way: 'mouse' + 'mouse_id (i.e.
a number)'
@author: penna
"""

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn import cross_decomposition
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from scipy.spatial import distance
import seaborn as sns


def reading(file_in, delimiter):

    """
    Function that reads csv file, deletes not social contact behaviours and returns a z-scored dataframe.

    :param file_in: path of csv file
    :param delimiter: delimiter
    :return: a df that contains the z-scored values
    """

    data_frame = pd.read_csv(file_in, sep=delimiter)
    data_frame = data_frame.drop('Frames', 1)
    list_index_to_delete = []
    neurons_list = []
    beh = ['defense action', 'Attack', 'face the consp', 'Sniff', 'Upright', 'Sniff A/G', 'domination hands on']
    for i in range(len(data_frame)):
        if data_frame['Beh'][i] not in beh:
            list_index_to_delete.append(i)
    data_frame = data_frame.drop('Beh', 1)
    for el in data_frame:
        neurons_list.append(el)
    data_frame = pd.DataFrame(StandardScaler().fit_transform(data_frame), columns=neurons_list)
    data_frame = data_frame.drop(data_frame.index[list_index_to_delete])
    return data_frame


def dataframe_merger_by_day(code_map, min_len):

    """
    Function that merges all the dataframes of the same day together.

    :param code_map: dictionary containing all dataframes in the format {day: list of dfs of that day}.
    :return: tupla of a list of dataframe and a list of labels
    """

    dataframes = []
    activities = []
    for code, lists in sorted(code_map.items()):
        temp_df = pd.DataFrame()
        for df in lists:
            df = (df.transpose().drop(df.transpose().index[min_len:])).transpose()
            if len(temp_df) == 0:
                temp_df = df
            else:
                index = df.index.tolist()
                for i in range(len(index)):
                    index[i] = i + len(temp_df) + 1
                df.index = index
                temp_df = temp_df.append(df)
        dataframes.append(temp_df)
        activities.append(code)
    return dataframes, activities


def dataframe_merger_and_column_adder(dataframes, activities, column_name):

    """
    Function that  adds the df of each experimental day together, and adds a column that identifies rows by day.

    :param dataframes: the list with all dataframes
    :param activities: the list containing days of experimental conditions
    :param column_name: name of the new column
    :return: a tupla containing the new dataframe and a list of all columns of the new dataframe that represent neurons
    """

    new_df = pd.DataFrame()
    for i in range(len(activities)):
        dataframes[i][column_name] = activities[i]
        new_df = new_df.append(dataframes[i])
        new_df.index = range(0, len(new_df))
    neurons = list(new_df.columns)
    neurons.remove(column_name)
    return new_df, neurons


def lda(df, df2):

    """
    Function that given a dataframe of N components and a series of labels, returns a dataframe of two components
    obtained applying the Linear Discriminant Analysis.

    :param df: first dataframe containing the data
    :param df2: second dataframe containing the labels (series)
    :return: a dataframe obtained after the dimensionality reduction method (LDA)
    """

    sklearn_lda = LDA(n_components=2)
    x_lda = sklearn_lda.fit_transform(df, df2)
    df_after_lda = pd.DataFrame(data=x_lda, columns=['lda 1', 'lda 2'])
    df_after_lda['target'] = df2
    return df_after_lda


def findAndChangeBehColumnPos(df):

    """
    Function that changes the position of the column that contains the behaviours recorded during the experiment,
    in order to keep the same convention in all files that are analysed.

    :param df: a dataframe with 'Beh' column in the last position
    :return: a dataframe whit 'Beh' column in the first position
    """

    new_df = pd.DataFrame()
    ListColumn = df.columns.tolist()
    ListColumn = ListColumn[-1:] + ListColumn[: -1]
    new_df = df[ListColumn]
    return new_df


def FindDistance(df1, df2):

    """
    Function that finds the averaged distance between the elements of two dataframes.

    :param df1: first dataframe
    :param df2: second dataframe
    :return: averaged distance between df1 and df2
    """

    d = 0
    count = 0
    distances = []
    for i in range(len(df1)):
        for j in range(len(df2)):
            d = d + distance.euclidean((df1['lda 1'][i], df1['lda 2'][i]), (df2['lda 1'][j], df2['lda 2'][j]))
            count = count + 1
        d2 = d / count
        distances.append(d2)
        d = 0
        count = 0
    dist = sum(distances) / len(distances)
    return dist


def create_df_for_hm(list_dist, list_column_name):

    """
    Function that creates a dataframe containing the distance between each cluster of experimental day; this dataframe
    is then used to create an heatmap that allows to visualize the distance between each cluster.

     :param list_dist: list that contains the distances between each cluster of experimental day (agg1 vs agg2,
            agg1 vs agg3, agg1 vs sf1, agg1 vs sf2, agg2 vs agg3, agg2 vs sf1, agg2 vs sf2, agg3 vs sf1, agg3 vs sf2,
            sf1 vs sf2)
    :param list_column_name: columns name for the dataframe (SF1, SF2, A1, A2, A3)
    :return: a df that in each cells contains the distance between each cluster of position, and has the same columns and
             the same indexes
    """

    df = pd.DataFrame(index=['SF1', 'SF2', 'A1', 'A2', 'A3'])
    for i in range(len(list_dist)):
        df[list_column_name[i]] = list_dist[i]
    return df


def heatmap_of_distance_creation(dataframe_after_lda, mouseID):

    """
    Function that creates for each mouse recorded an heatmap that allows to visualize the distance between each cluster of
    day.This function saves the hm into a folder defined by an absolute path

    :param dataframe_after_lda: dataframe after the dimensionality reduction method(LDA)
    :param mouse_id: number of mouse
    :return: a list of list containing the distances between each cluster of position
    """

    n = mouseID
    df_a1 = dataframe_after_lda.loc[dataframe_after_lda['target'] == 'A1']
    df_a2 = dataframe_after_lda.loc[dataframe_after_lda['target'] == 'A2']
    df_a2.index = range(0, len(df_a2))
    df_a3 = dataframe_after_lda.loc[dataframe_after_lda['target'] == 'A3']
    df_a3.index = range(0, len(df_a3))
    df_sf1 = dataframe_after_lda.loc[dataframe_after_lda['target'] == 'F1']
    df_sf1.index = range(0, len(df_sf1))
    df_sf2 = dataframe_after_lda.loc[dataframe_after_lda['target'] == 'F2']
    df_sf2.index = range(0, len(df_sf2))
    dd_a12 = FindDistance(df_a1, df_a2)
    dd_a13 = FindDistance(df_a1, df_a3)
    dd_a1_sf1 = FindDistance(df_a1, df_sf1)
    dd_a1_sf2 = FindDistance(df_a1, df_sf2)
    l_a1 = (dd_a1_sf1, dd_a1_sf2, 0, dd_a12, dd_a13)
    dd_a23 = FindDistance(df_a2, df_a3)
    dd_a2_sf1 = FindDistance(df_a2, df_sf1)
    dd_a2_sf2 = FindDistance(df_a2, df_sf2)
    l_a2 = (dd_a2_sf1, dd_a2_sf2, dd_a12, 0, dd_a23)
    dd_a3_sf1 = FindDistance(df_a3, df_sf1)
    dd_a3_sf2 = FindDistance(df_a3, df_sf2)
    l_a3 = (dd_a3_sf1, dd_a3_sf2, dd_a13, dd_a23, 0)
    dd_sf1_sf2 = FindDistance(df_sf1, df_sf2)
    l_sf1 = (0, dd_sf1_sf2, dd_a1_sf1, dd_a2_sf1, dd_a3_sf1)
    l_sf2 = (dd_sf1_sf2, 0, dd_a1_sf2, dd_a2_sf2, dd_a3_sf2)
    list_of_dist = [l_sf1, l_sf2, l_a1, l_a2, l_a3]
    list_name = ['SF1', 'SF2', 'A1', 'A2', 'A3']
    df_hm = create_df_for_hm(list_of_dist, list_name)
    plt.subplot(231)
    ax = plt.axes()
    cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
    mask = np.zeros_like(df_hm, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    np.fill_diagonal(mask, 0)
    sm = sns.heatmap(df_hm, mask=mask, annot=True, vmin=2, vmax=6, cmap=cmap)
    ax.set_title('Mouse' + n + ' Distance HeatMap')
    figure = sm.get_figure()
    figure.savefig(r'C:\Users\penna\Desktop' + '\Mouse' + n + 'DistanceHeatMap2',
                   dpi=400)
    return list_of_dist


def main_function(mouse_id):

    """
    Function that recalls the functions described above: reads the files, calculates the z-scored values,
    and merges recordings of the same days together. It creates an unique dataframe merging all the experimental days
    and on this it applies the LDA.
    Then finds the distance between couple of experimental days for each mouse in input and the averaged distance.
    Finally it creates the heat_maps of the distances and saves them in a folder specified by an absolute path.

    :param mouse_id: number that identifies the mouse that we want to analyse
    :return: the hm of distance between each experimental day for each mouse and the hm of distances obtained
    averaging all mice given in input
    """

    all_dist = []
    for direct in os.listdir(r'C:\Users\penna\Desktop\aggression-social fear files'):
        minLen = 999999999999
        codeMap = {}
        for n in mouse_id:
            if direct == "mouse" + n:
                for file in os.listdir(
                        r'C:\Users\penna\Desktop\aggression-social fear files' + "\\" + direct):
                    file = r'C:\Users\penna\Desktop\aggression-social fear files' + "\\" + direct + "\\" + file
                    data = reading(file, ';')
                    file_code = file[-10] + file[-9]
                    file_code = file_code.upper()
                    minLen = min(len(data.transpose()), minLen)
                    if file_code not in codeMap.keys():
                        codeMap.update({file_code: [data]})
                    else:
                        codeMap.update({file_code: codeMap.get(file_code) + [data]})
            else:
                continue
            data_frame_list, activity_list = dataframe_merger_by_day(codeMap, minLen)
            dfs = data_frame_list
            Data, features = dataframe_merger_and_column_adder(dfs, activity_list, 'target')
            Label = Data.loc[:, 'target']
            Data = Data.drop('target', 1)
            x = Data
            y = Label
            df_lda = lda(x, y)
            all_dist.append(heatmap_of_distance_creation(df_lda, n))
    # for averaged distance
    res = []
    for x in range(len(all_dist[0])):
        dist = []
        for y in range(len(all_dist[0][0])):
            dist.append(0)
        res.append(dist)

    for mouse in all_dist:
        for gg in range(len(mouse)):
            for i in range(len(mouse[gg])):
                res[gg][i] += mouse[gg][i]
    for gg in range(len(res)):
        for val in range(len(res[gg])):
            res[gg][val] = res[gg][val] / len(all_dist)
    dfHm = create_df_for_hm(res, ['SF1', 'SF2', 'A1', 'A2', 'A3'])
    plt.subplot(231)
    ax = plt.axes()
    cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
    mask = np.zeros_like(dfHm, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    np.fill_diagonal(mask, 0)
    sm = sns.heatmap(dfHm, annot=True, mask=mask, cmap=cmap, vmin=2, vmax=10)
    ax.set_title('ALLMiceMean' + ' Distance HeatMap')
    figure = sm.get_figure()
    figure.savefig('C:/Users/Krzywy/Desktop/Calcium analysis/graphs/Heatmaps/LDAstandardMean', dpi=400)
    return


''' script'''
main_function(["4", "5", "8", "10"])
