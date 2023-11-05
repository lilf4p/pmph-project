# import csv file and plot it 
# choose metric to plot -> then only one value for the other metrics can be plotted 
# y axis will be always performance

import pandas as pd
import matplotlib.pyplot as plt

def plot_blocks(input_size:int):

    df = pd.read_csv('benchmarks/bench-sps-15.csv')

    df_baseline = pd.read_csv('benchmarks/bench-naiveMemcpy-15.csv')

    # remove the entry with bandwidth equal -1
    df = df[df['bandwidth'] != -1]

    # take only entry with input 
    df = df[df['input'] == input_size]
    df_baseline = df_baseline[df_baseline['input'] == input_size]
    #remove baseline duplicates lines
    df_baseline = df_baseline.drop_duplicates(subset=['block'])
    # remove 1024 block from baseline 
    df_baseline = df_baseline[df_baseline['block'] != 1024]

    print(df)
    print(df_baseline)

    # set matplotlib style to ggplot
    plt.style.use('ggplot')

    # plot on same graph the baseline with dotted line light gray
    ax = df_baseline.plot(x='block', y='bandwidth', linestyle='--', color='darkgray')

    # plot the kernel 4
    df_kernel4 = df[df['kernel'] == 4]
    df_kernel4.plot(ax = ax, x='block', y='bandwidth', marker='o')

    # plot kernel 3
    df_kernel3 = df[df['kernel'] == 3]
    df_kernel3.plot(ax = ax, x='block', y='bandwidth', marker='o')

    # plot only if kernel is equal 2
    df_kernel2  = df[df['kernel'] == 2]
    df_kernel2.plot(ax = ax, x='block', y='bandwidth', marker='o')
    
    # plot only if kernel is equal 2
    df_kernel1  = df[df['kernel'] == 1]
    df_kernel1.plot(ax = ax, x='block', y='bandwidth', marker='o')

    plt.legend(['Naive Memcpy', 'LB Optimized', 'LB Warp', 'LB Single Thread', 'Without LB'], fontsize='medium')

    # add axis name 
    plt.xlabel('Block size', weight = 'bold')
    plt.ylabel('Bandwidth (GB/s)', weight = 'bold')

    plt.xticks(weight = 'bold')
    plt.yticks(weight = 'bold')

    # set the x axis tick manually 
    plt.xticks([64,128,256,512])

    plt.savefig('plots/sps-block-15-'+str(input_size)+'.png')

    plt.show()

# do the same for chunk sizes over performance 
def plot_chunks(input_size:int):
    
    block = 512 #fixed
    # input sizes 221184, 1000000, 100003565

    df = pd.DataFrame()
    df_baseline = pd.DataFrame()

    for chunk in [1,2,6,8,10,12,14,15]:
         
        df_tmp = pd.read_csv('benchmarks/bench-sps-'+str(chunk)+'.csv')
        df_baseline_tmp = pd.read_csv('benchmarks/bench-naiveMemcpy-'+str(chunk)+'.csv')
        
        # add to df only the df_tmp entry with input size = input_size and block = 512
        df_tmp = df_tmp[(df_tmp['input'] == input_size) & (df_tmp['block'] == block)]
        df = df._append(df_tmp)

        # add to df only the df_tmp entry with input size = input_size and block = 512
        df_baseline_tmp = df_baseline_tmp[(df_baseline_tmp['input'] == input_size) & (df_baseline_tmp['block'] == block)]
        df_baseline = df_baseline._append(df_baseline_tmp)

    # remove the entry with bandwidth equal -1
    df = df[df['bandwidth'] != -1]

    # compact baseline in a single entry with the average bandwiwdth
    df_baseline = df_baseline.groupby(['input']).mean()

    print(df)
    print(df_baseline)

    # set matplotlib style to ggplot
    plt.style.use('ggplot')

    baseline_value = df_baseline['bandwidth'].item()
    print(baseline_value)
    
    df_kernel4 = df[df['kernel'] == 4]
    ax = df_kernel4.plot(x='chunk', y='bandwidth',marker='o', label='LB Optimized')

    df_kernel3 = df[df['kernel'] == 3]
    df_kernel3.plot(ax = ax, x='chunk', y='bandwidth', marker='o', label='LB Warp')
    
    df_kernel2  = df[df['kernel'] == 2]
    df_kernel2.plot(ax = ax, x='chunk', y='bandwidth', marker='o', label='LB Thread')

    df_kernel1  = df[df['kernel'] == 1]
    df_kernel1.plot(ax = ax, x='chunk', y='bandwidth', marker='o', label='Without LB')

    plt.axhline(y=baseline_value, color='darkgray', linestyle='--', label='Naive Memcpy')

    # change legend order, so that the first one is the baseline
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[4], handles[0], handles[1], handles[2], handles[3]]
    labels = [labels[4], labels[0], labels[1], labels[2], labels[3]]
    ax.legend(handles, labels, fontsize='medium')


    #plt.legend(['Naive Memcpy', 'LB Single Thread', 'LB Warp', 'LB Optimized'], fontsize='medium')

    # add axis name 
    plt.xlabel('Chunk size', weight = 'bold')
    plt.ylabel('Bandwidth (GB/s)', weight = 'bold')

    plt.xticks(weight = 'bold')
    plt.yticks(weight = 'bold')

    # set the x axis tick manually 
    plt.xticks([1,2,4,6,8,10,12,14,15])

    plt.savefig('plots/sps-chunk-512-'+str(input_size)+'.png')

if __name__ == '__main__':
    #plot_blocks(221184)
    plot_chunks(221184)
    plot_chunks(1000000)
    plot_chunks(100003565)
    plot_blocks(221184)
    plot_blocks(1000000)
    plot_blocks(100003565)
