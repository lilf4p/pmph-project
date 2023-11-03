# import csv file and plot it 
import pandas as pd
import matplotlib.pyplot as plt

def plot(metric:str = 'input'):

    df = pd.read_csv('bench-sps-12.csv')

    # remove the entry with bandwidth equal -1
    df = df[df['bandwidth'] != -1]

    # take only entry with block == 512
    if metric == 'input': df = df[df['block'] == 512]
    elif metric == 'block': df = df[df['input'] == 100003565]

    # take only columns input and bandwidth
    df = df[['kernel', metric, 'bandwidth']]

    print(df)

    # set matplotlib style to ggplot
    plt.style.use('ggplot')

    # plot only if kernel is equal 2
    df_kernel2  = df[df['kernel'] == 2]
    ax = df_kernel2.plot(x=metric, y='bandwidth', color='blue')

    # plot on the same graph of kernel 2
    df_kernel3 = df[df['kernel'] == 3]
    df_kernel3.plot(ax = ax, x=metric, y='bandwidth', color='red')

    plt.legend(['Kernel v2', 'Kernel v3'])

    # add axis name 
    if metric == 'input': plt.xlabel('Input', weight = 'bold')
    elif metric == 'block': plt.xlabel('Block', weight = 'bold')
    plt.ylabel('Bandwidth (GB/s)', weight = 'bold')

    plt.show()

if __name__ == '__main__':
    plot('block')
