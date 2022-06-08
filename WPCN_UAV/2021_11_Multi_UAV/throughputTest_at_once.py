import throughputTest as TT
import throughputTest_Genetic as TTG
import throughputTest_Additional as TTA
import helper as h_
import numpy as np
import pandas as pd

if __name__ == '__main__':

    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('always')
    warnings.simplefilter('ignore')

    # numpy setting
    np.set_printoptions(edgeitems=20, linewidth=170)

    # load settings
    paperArgs = h_.loadSettings({'fc':'float', 'ng':'float', 'B':'float', 'o2':'float',
                                'b1':'float', 'b2':'float',
                                'alphaP':'float', 'alphaL':'float',
                                'mu1':'float', 'mu2':'float',
                                's':'float', 'PD':'float', 'PU':'float',
                                'width':'float', 'height':'float',
                                'M':'int', 'L':'int', 'devices':'int', 'T':'float', 'N':'int', 'H':'float',
                                'iters':'int',
                                'clusteringAtLeast':'float', 'clusteringAtMost':'float',
                                'windowSize':'int',
                                'epochs':'int',
                                'isStatic':'logical'})

    fc = paperArgs['fc']
    ng = paperArgs['ng']
    B = paperArgs['B']
    o2 = paperArgs['o2']
    b1 = paperArgs['b1']
    b2 = paperArgs['b2']
    alphaP = paperArgs['alphaP']
    alphaL = paperArgs['alphaL']
    mu1 = paperArgs['mu1']
    mu2 = paperArgs['mu2']
    s = paperArgs['s']
    PU = paperArgs['PU']
    width = paperArgs['width']
    height = paperArgs['height']
    M = paperArgs['M']
    L = paperArgs['L']
    devices = paperArgs['devices']
    T = paperArgs['T']
    N = paperArgs['N']
    H = paperArgs['H']
    iters = paperArgs['iters']
    clusteringAtLeast = paperArgs['clusteringAtLeast']
    clusteringAtMost = paperArgs['clusteringAtMost']
    windowSize = paperArgs['windowSize']
    epochs = paperArgs['epochs']
    isStatic = paperArgs['isStatic']

    # manual setting: iters   (the number of iterations),
    #                 L       (the number of UAVs = clusters),
    #                 devices (the number of devices),
    #             and N       (the number of time slots)
    iters   = 100
    L       = 5
    devices = 20
    Ns      = [35]

    # execute
    for N in Ns:

        # list of min throughputs
        minThroughputList = []

        input_data = []
        output_data = []

        # execute THROUGHPUT TEST
        for iterationCount in range(iters):
            print('\n**** iteration ' + str(iterationCount) + ' ****\n')
            
            TTA.throughputTest(M, T, N, L, devices, width, height, H,
                               ng, fc, B, o2, b1, b2, alphaP, None, mu1, mu2, s, None, PU,
                               iterationCount, minThroughputList, clusteringAtLeast, clusteringAtMost,
                               input_data, output_data, True, None, windowSize, isStatic)

        # save min throughput list as *.csv file
        minThroughputList = pd.DataFrame(np.array(minThroughputList))
        minThroughputList.to_csv('minThroughputList_iter_' + ('%04d' % iters) + '_L_' + ('%04d' % L) +
                                 '_devs_' + ('%04d' % devices) + '_N_' + ('%04d' % N) + '.csv')

        # save min throughput list as *.txt file
        arr = np.array(minThroughputList)[:, 1:]
        note = 'mean: ' + str(np.mean(arr)) + ', std: ' + str(np.std(arr)) + ', nonzero: ' + str(np.count_nonzero(arr))

        noteFile = open('minThroughputList_iter_' + ('%04d' % iters) + '_L_' + ('%04d' % L) +
                                 '_devs_' + ('%04d' % devices) + '_N_' + ('%04d' % N) + '.txt', 'w')
        
        noteFile.write(note)
        noteFile.close()

    # get and train model
    if isStatic == False:
        print('\n **** NOT STATIC -> train and test the model !! ****\n')
        
        try:
            model = tf.keras.models.load_model('WPCN_UAV_DL_model')
            print('model load succeeded')
        except:
            print('model load failed')
            model = TTA.getAndTrainModel(epochs, windowSize)

        # run test (using 5% (min 10) iterations of training)
        iters = max(10, iters // 20)

        # input and output data for testing
        test_input_data = []
        test_output_data = []

        for iterationCount in range(iters):
            print('TEST ITER COUNT ', iterationCount, '/', iters)

            throughputTest(M, T, N, L, devices, width, height, H,
                           ng, fc, B, o2, b1, b2, alphaP, None, mu1, mu2, s, None, PU,
                           iterationCount, minThroughputList, clusteringAtLeast, clusteringAtMost,
                           test_input_data, test_output_data, False, model, windowSize, False)
