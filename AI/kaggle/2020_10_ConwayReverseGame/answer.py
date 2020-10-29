import sys
import numpy as np
sys.path.insert(0, '../../../../AI_BASE')

import readData as RD

if __name__ == '__main__':
    train = RD.loadArray('train.csv', ',')
    test = RD.loadArray('test.csv', ',')

    print(np.array(train))
    print(np.array(test))

    
