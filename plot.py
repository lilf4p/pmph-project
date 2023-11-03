# import csv file and plot it 
# choose metric to plot -> then only one value for the other metrics can be plotted 
# y axis will be always performance

import pandas as pd
import matplotlib.pyplot as plt

def plot(metric:str = 'input'):

    df = pd.read_csv('benchmarks/bench-sps-15.csv')

    # remove the entry with bandwidth equal -1
    df = df[df['bandwidth'] != -1]

    # take only entry with block == 512
    if metric == 'input': df = df[df['block'] == 512]
    elif metric == 'block': df = df[df['input'] == 100003565]

    # take the first value of row chunk 
    chunk = df['chunk'].tolist()[0]


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

    if metric == 'input': plt.title('Bandwidth vs ' + metric + ' with chunk='+str(chunk)+' and block size=512')
    elif metric == 'block': plt.title('Bandwidth vs ' + metric + ' with chunk='+str(chunk)+' and input=100003565')

    plt.legend(['Kernel v2', 'Kernel v3'])

    # add axis name 
    if metric == 'input': plt.xlabel('Input', weight = 'bold')
    elif metric == 'block': plt.xlabel('Block', weight = 'bold')
    plt.ylabel('Bandwidth (GB/s)', weight = 'bold')

    plt.show()

if __name__ == '__main__':
    plot('block')
    plot('input')
