# import csv file and plot it 
# choose metric to plot -> then only one value for the other metrics can be plotted 
# y axis will be always performance

import pandas as pd
import matplotlib.pyplot as plt

def plot(metric:str = 'input', chunk:int=15, fixed:int = 512):

    df = pd.read_csv('benchmarks/bench-sps-'+str(chunk)+'.csv')

    # remove the entry with bandwidth equal -1
    df = df[df['bandwidth'] != -1]

    # take only entry with block == 512
    if metric == 'input': df = df[df['block'] == fixed]
    elif metric == 'block': df = df[df['input'] == fixed]

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

    # plot on same graph the kernel 4
    df_kernel4 = df[df['kernel'] == 4]
    df_kernel4.plot(ax = ax, x=metric, y='bandwidth', color='green')

    if metric == 'input': plt.title('Performances over ' + metric + ' sizes \n with chunk='+str(chunk)+' and block size='+str(fixed), fontsize=13, weight='bold', pad=15)
    elif metric == 'block': plt.title('Performances over ' + metric + ' sizes \n with chunk='+str(chunk)+' and input size='+str(fixed), fontsize=13, weight='bold', pad=15)

    plt.legend(['Thread', 'Warp', 'Optimized'], title='SPS Kernel', loc=4, fontsize='medium')

    # add axis name 
    if metric == 'input': plt.xlabel('Input sizes', weight = 'bold')
    elif metric == 'block': plt.xlabel('Block sizes', weight = 'bold')
    plt.ylabel('Bandwidth (GB/s)', weight = 'bold')

    plt.xticks(weight = 'bold')
    plt.yticks(weight = 'bold')

    plt.show()

if __name__ == '__main__':
    plot('block',15, 100003565)
    plot('input',15, 512)
