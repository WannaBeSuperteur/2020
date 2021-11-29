import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    
    # load file list
    fileList = os.listdir()
    
    # read each *.txt file and make array (logData) for data
    logData = []
    
    for file in fileList:
        if file.startswith('minThroughputList_iter_') and file.endswith('.txt'):
            f = open(file, 'r')
            fdata = f.readlines()[0]
            f.close()

            data_iters   = int(file.split('.')[0].split('_')[-1])
            data_mean    = float(fdata.split(': ')[1].split(',')[0])
            data_std     = float(fdata.split(': ')[2].split(',')[0])
            data_nonzero = int(fdata.split(': ')[3])

            logData.append([data_iters, data_mean, data_std, data_nonzero])

    # convert into np.array
    logData = np.array(logData)
    print(logData)

    # draw and display graph
    plt.clf()
            
    fig, ax = plt.subplots()
    n       = len(logData)

    iters   = logData[:, 0]
    mean    = logData[:, 1] / max(logData[:, 1])
    std     = logData[:, 2] / max(logData[:, 2])
    nonzero = logData[:, 3] / max(logData[:, 3])

    plt.clf()
    plt.suptitle('min throughput log (relative)')
    fig.text(0.3, 0.9, 'mean', color='r')
    fig.text(0.45, 0.9, 'std', color='y')
    fig.text(0.6, 0.9, 'nonzero', color='b')
            
    for i in range(n):
        plt.scatter(iters[i], mean[i], s=30, c='r')
        plt.scatter(iters[i], std[i], s=30, c='y')
        plt.scatter(iters[i], nonzero[i], s=30, c='b')

        if i < n-1:
            plt.plot([iters[i], iters[i+1]], [mean[i], mean[i+1]], linewidth=2, c='r')
            plt.plot([iters[i], iters[i+1]], [std[i], std[i+1]], linewidth=2, c='y')
            plt.plot([iters[i], iters[i+1]], [nonzero[i], nonzero[i+1]], linewidth=2, c='b')
                    
    plt.axis([0, max(logData[:, 0])*1.025, -0.05, 1.05])
    plt.savefig('minThroughputLog.png')
