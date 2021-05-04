import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np
import read_val_report

def useTestOutput(fn, threshold):
    
    # read file
    testResult = RD.loadArray(fn)

    # write final result
    finalResult = []
    
    for i in range(len(testResult)):
        value = float(testResult[i][0])
        
        if threshold == None:
            finalResult.append([value])
        else:
            if value < threshold:
                finalResult.append([0])
            else:
                finalResult.append([1])

    # write file
    RD.saveArray('to_submit.txt', finalResult)

def useAdvancedModels():
    count_lightGBM = 1
    count_DecisionTree = 1
    count_deepLearning = 0

    pred_array = read_val_report.getPredAndRealArray(count_lightGBM,
                                                     count_DecisionTree,
                                                     count_deepLearning,
                                                     None)

    RD.saveArray('final_test_output.txt', np.array([pred_array]).T)

if __name__ == '__main__':         
    useAdvancedModels()
    useTestOutput('final_test_output.txt', 0.75)
