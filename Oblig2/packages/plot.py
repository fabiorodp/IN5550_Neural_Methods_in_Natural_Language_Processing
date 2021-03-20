from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # needed for 3d plots
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import pandas as pd
import ArgumentPaser

DIR, FILE = os.path.split(os.path.abspath(__file__))


# Static plot titles

def heatmap(header: str, subdir: str = 'output'):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4), sharey=False)
    fig.suptitle('Heatmap')
    all_data = get_heatmap_data(header, subdir)

    y, x = header_to_title(header)

    for i, title in enumerate(htm_titles()):
        data = all_data[i]
        sns.heatmap(
            ax=axes[i],
            data=data,
            annot=True,
            fmt=".3g",
        )
        axes[i].set_title(title)
        axes[i].set_xlabel(x)
        axes[i].set_ylabel(y)


def plot3d_time_accuracy(header: str, subdir: str = 'output'):
    X1, X2, time, _, _,  _, fsco = header_to_data(header, subdir)

    title1, title2 = header_to_title(header, subdir)

    # plot first param
    plot_3d(X1, time, fsco, title1, title1, 'Time')

    # plot second param
    plot_3d(X2, time, fsco, title2, title2, 'Time')


def plot3d_pars_time(header: str, subdir: str = 'output'):
    X1, X2, Z, _, _,  _, _ = header_to_data(header, subdir)

    title1, title2 = header_to_title(header, subdir)

    # plot first param
    plot_3d(X1, X2, Z, title1+' vs '+title2+' vs Time', title1, title2)


def plot3d_pars_accuracy(header: str, subdir: str = 'output'):
    X1, X2, _, Z, _,  _, _ = header_to_data(header, subdir)

    title1, title2 = header_to_title(header, subdir)

    # plot first param
    plot_3d(X1, X2, Z, title1+' vs '+title2+' vs Accuracy', title1, title2)


def plot3d_pars_fscore(header: str, subdir: str = 'output'):
    X1, X2, _, _, _,  _, Z = header_to_data(header, subdir)

    title1, title2 = header_to_title(header, subdir)

    # plot first param
    plot_3d(X1, X2, Z, title1+' vs '+title2+' vs F-Score', title1, title2)


def plot_all():
    header = [
        'ActFunct',
        'BoW',
        'EpochsBacthes',
        'hlay',
        'Mmt'
    ]

# helper functions
def header_to_title(header, subdir: str = 'output'):
    title1 = ''
    title2 = ''
    if 'act' in header.lower():
        title1 = 'Hidden Activation Functions'
        title2 = 'Output Activation Functions'
    if 'bat' in header.lower():
        title1 = 'Epochs'
        title2 = 'Batch size'
    if 'bow' in header.lower():
        title1 = 'Vocab Size'
        title2 = 'Bag of Words'
    if 'hl' in header.lower():  
        title1 = 'Hidden Layers'
        title2 = 'Units'
    if 'mmt' in header.lower():
        title1 = 'Learning Rate'
        title2 = 'Momentum'

    return title1, title2


def header_to_data(header, subdir: str = 'output'):
    '''
    Return:
        parameter1
        parameter2
        time
        accuracy
        precision
        recall
        fscore
    '''
    time_file = header + '_0.csv'
    accu_file = header + '_1.csv'
    prec_file = header + '_2.csv'
    reca_file = header + '_3.csv'
    fsco_file = header + '_4.csv'

    # str to path
    pth1 = os.path.join(DIR, subdir, time_file)
    pth2 = os.path.join(DIR, subdir, accu_file)
    pth3 = os.path.join(DIR, subdir, prec_file)
    pth4 = os.path.join(DIR, subdir, reca_file)
    pth5 = os.path.join(DIR, subdir, fsco_file)

    # load data
    time = pd.read_csv(pth1, header=0, index_col=0)
    accu = pd.read_csv(pth2, header=0, index_col=0)
    prec = pd.read_csv(pth3, header=0, index_col=0)
    reca = pd.read_csv(pth4, header=0, index_col=0)
    fsco = pd.read_csv(pth5, header=0, index_col=0)

    # these paraemters
    par1 = time.index.astype('float')
    par2 = time.columns.astype('float')
    par1, par2 = np.meshgrid(par1, par2)

    return par1.T, par2.T, time, accu, prec, reca, fsco


def htm_titles():
    return ["Training elapsed time in seconds",
            "Accuracy Score Heatmap",
            "Macro F1 Score Heatmap"]


def get_heatmap_data(header: str, subdir: str = 'output'):
    all_data = []
    for i in [0, 1, 4]:
        filename = os.path.join(DIR, subdir, header+f'_{i}.csv')
        data = pd.read_csv(filename, header=0, index_col=0)
        all_data.append(data)
    return all_data


def plot_3d(X, Y, Z, title, xlabel, ylabel):
    # setup the figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # map surface to axis
    surf = ax.plot_surface(X, Y, Z,
                           cmap=cm.coolwarm,
                           linewidth=1,
                           )

    # config figure
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def show():
    plt.show()


if __name__ == '__main__':

    header = 'lrm'
    subdir = 'output'

    # command line args: header subdir
    if len(sys.argv) > 1:
        header = sys.argv[1]
        if len(sys.argv) > 2:
            subdir = sys.argv[2]

    print(f'Showing plots for {header_to_title(header, subdir)}')

    # heatmap
    heatmap(header, subdir)
    show()

    # 3d surface plot with x:parameter, y:time, z:accuracy
    plot3d_time_accuracy(header, subdir)

    # 3d surface plot with x:parameter1, y:parameter2, z:time
    plot3d_pars_time(header, subdir)  

    # 3d surface plot with x:parameter1, y:parameter2, z:accuracy
    plot3d_pars_accuracy(header, subdir)  

    # 3d surface plot with x:parameter1, y:parameter2, z:fscore
    plot3d_pars_fscore(header, subdir)  

    show()
    