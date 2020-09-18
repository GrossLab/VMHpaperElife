# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 11:45:57 2018

This script allows to calculate and visualize the Linear Discriminant Analysis applied on csv file recorded during
social fear and aggression experiments. First, it reads each recordings, finds the z-scored values, and merges the
recordings that belong to the same experimental day together.
Then merges the experimental days in an unique dataset and creates a label that allows to know each row of this dataset
during which day was recorded. After that it calculates the LDA that allows to check if there are cluster of frames where
the neuronal activity is similar.
The csv file has to follow a specific name convention: 'Vmh' + 'id_numb of mouse' + 'a string (A or SF )' + 'a numb that
identifies the day of recording' + 'a number that identifies the number of recording' + 'a string (beh)' 23A10beh.

@author: penna
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
import matplotlib.patches as patches


def reading(file_in, delimiter):

    """
    Function that reads from csv file, deletes not social contact behaviours and returns a z-scored df.

    :param file_in: absolute file path of the file csv
    :param delimiter: delimiter of the csv file
    :return: a Dataframe that contains the Z-scored df/f values
    """

    data_frame = pd.read_csv(file_in, sep=delimiter)
    data_frame = data_frame.drop('Frames', 1)
    indexes_to_del = []
    neurons = []
    beh = ['Attack', 'defense action', 'face the consp', 'Sniff', 'Upright', 'Sniff A/G', 'domination hands on']
    for i in range(len(data_frame)):
        if data_frame['Beh'][i] not in beh:
            indexes_to_del.append(i)
    data_frame = data_frame.drop('Beh', 1)
    for el in data_frame:
        neurons.append(el)
    data_frame = pd.DataFrame(StandardScaler().fit_transform(data_frame), columns=neurons).drop(data_frame.index[indexes_to_del])
    return data_frame


def dataframe_merger_by_day(code_map):

    """
    Function that merges all dataframes of the same day together.

    :param code_map: dictionary containing all dataframes in the format {day: list of dfs of that day}.
    :return: tupla of a list of dataframe and a list of labels
    """

    dataframes = []
    activities = []
    for code, lists in sorted(code_map.items()):
        temp_df = pd.DataFrame()
        for df in lists:
            df = (df.transpose().drop(df.transpose().index[minLen:])).transpose()
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
    Function that  adds all new_df of each experimental day together, and adds a column that identifies rows by day.

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


def graphing(df_after_LDA, title, days, colours, days_column, variance):

    """
    Functions that creates graph of reduction dimensionality method.

    :param df_after_LDA: the dataframe after the dimensionality reduction
    :param title: figure title
    :param days: the list containing days of experimental conditions
    :param colours: the list containing colours for the legend
    :param days_column: name of the column that defines the day of each row
    :param variance: explained variance of each LDA component
    :return: a matplotlib.figure.Figure of LDA dimensionality reduction
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xlabel('LDA 1', fontsize=10)
    ax.set_ylabel('LDA 2', fontsize=10)
    ax.set_title(title, fontsize=15)
    ax.set_xlim([-6, 8])
    ax.set_ylim([-7, 8])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.grid(b=None)

    targets = days
    for target, color in zip(targets, colours):

        if target == 'A1':
            indices_to_keep = df_after_LDA[days_column] == target
            ax.scatter(df_after_LDA.loc[indices_to_keep, 0]
                       , df_after_LDA.loc[indices_to_keep, 1]
                       , c=color
                       , s=10
                       , alpha=1)
        elif target == 'A2':
            indices_to_keep = df_after_LDA[days_column] == target
            ax.scatter(df_after_LDA.loc[indices_to_keep, 0]
                       , df_after_LDA.loc[indices_to_keep, 1]
                       , c=color
                       , s=10
                       , alpha=1)
        elif target == 'A3':
            indices_to_keep = df_after_LDA[days_column] == target
            ax.scatter(df_after_LDA.loc[indices_to_keep, 0]
                       , df_after_LDA.loc[indices_to_keep, 1]
                       , c=color
                       , s=10
                       , alpha=1)
        elif target == 'F1':
            indices_to_keep = df_after_LDA[days_column] == target
            ax.scatter(df_after_LDA.loc[indices_to_keep, 0]
                       , df_after_LDA.loc[indices_to_keep, 1]
                       , c=color
                       , s=10
                       , alpha=1)
        else:
            indices_to_keep = df_after_LDA[days_column] == target
            ax.scatter(df_after_LDA.loc[indices_to_keep, 0]
                       , df_after_LDA.loc[indices_to_keep, 1]
                       , c=color
                       , s=10
                       , alpha=1)
    label = ['LDA1', 'LDA2', 'LDA3', 'LDA4']
    ax.grid()
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_xlabel('LDA Components', fontsize=10)
    ax2.set_ylabel('Variance Explained', fontsize=10)
    ax2.bar(label, variance)
    dic = {'Aggression1': 'lightblue', 'Aggression2': 'dodgerblue', 'Aggression3': 'navy', 'Defense1': 'lightcoral',
           'Defense2': 'red'}
    # dic =  {   'Defense1':'lightcoral', 'Defense2':'crimson'}
    # dic = {'Aggression1':'lightblue','Aggression2':'dodgerblue', 'Aggression3':'navy'}
    j = 1
    p = 0.09
    for beh in sorted(dic.keys()):
        legend_patch = patches.Patch(color=dic.get(beh), label=beh)
        legenda = plt.legend(handles=[legend_patch], loc=1, bbox_to_anchor=(1.12, p * 15), prop={'size': 10})
        p += 0.01
        if j != len(dic.keys()):
            j += 1
            ax = plt.gca().add_artist(legenda)
        else:
            plt.show()

            fig.savefig(r'D:\l' + title, transparent=True)
    return fig


''' From here we call all the functions created above '''

path = r'D:\Lavoro\roba per piotr\aggression-social fear files\mouse23'
filepaths = []
for files in os.listdir(path):
    files = str(files)
    filepaths.append(files)
minLen = 999999999999
codeMap = {}

''' To select only neurons present in each day and create a dictionary with the all dataframes '''

for file in filepaths:
    data = reading(path + "\\" + file, ';')
    fileCode = file[-10] + file[-9]
    fileCode = fileCode.upper()
    minLen = min(len(data.transpose()), minLen)
    if fileCode not in codeMap.keys():
        codeMap.update({fileCode: [data]})
    else:
        codeMap.update({fileCode: codeMap.get(fileCode) + [data]})
dataFrameList, activityList = dataframe_merger_by_day(codeMap)
normDfs = dataFrameList

''' Create one df with the all data for LDA'''

Data, features = dataframe_merger_and_column_adder(normDfs, activityList, 'target')
Label = Data.loc[:, 'target']
Data = Data.drop('target', 1)
x = Data
y = Label

''' LDA method '''

sklearn_lda = LDA(n_components=4)
X_lda = sklearn_lda.fit_transform(x, y)
var = sklearn_lda.explained_variance_ratio_
w = sklearn_lda.coef_

''' Graph part '''

df = pd.DataFrame(X_lda)
df['target'] = Label
colors = ['lightblue', 'dodgerblue', 'navy', 'lightcoral', 'crimson']
im = graphing(df, 'Defense', activityList, colors, 'target', var)
