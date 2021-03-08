import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np

if __name__ == '__main__':

    for num in [0, 1]:

        # read file
        testResult = RD.loadArray('test_output_' + str(num) + '.txt')

        # write final result
        finalResult = []
        
        for i in range(len(testResult)):

            if num == 0: # formation_energy_ev_natom
                finalResult.append([float(testResult[i][0]) * 0.104078 + 0.187614])

            else: # bandgap_energy_ev
                finalResult.append([float(testResult[i][0]) * 1.006635 + 2.077205])

        # write file
        RD.saveArray('to_submit_' + str(num) + '.txt', finalResult)
            
