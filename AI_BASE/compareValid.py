import math
import numpy as np
import readData as RD

# finalResult          : final result array
# validName            : validation data file name
#                        (training data file name, if validation data is from training data)
# validationCol        : column to validate, from validation data file
# trainValid_validRows : rows to validate
def compare(finalResult, validName, validationCol, trainValid_validRows):

    # assert that finalResult and validRows have the same length
    dataLen = len(finalResult) # length of data
    assert(dataLen == len(trainValid_validRows)) # assert the same length

    # read data
    validData = RD.loadArray(validName) # original validation data array
    validResult = [] # validation data array
    for i in range(dataLen):
        validResult.append(validData[trainValid_validRows[i]][validationCol])
    validResult = np.array(validResult)

    # compute MAE and MSE
    result = ''
    
    MAE = 0.0
    MSE = 0.0
    for i in range(dataLen):
        finalResult[i] = float(finalResult[i])
        validResult[i] = float(validResult[i])
        MAE += abs(float(finalResult[i]) - float(validResult[i]))
        MSE += pow(float(finalResult[i]) - float(validResult[i]), 2)

        result += 'pred=' + str(finalResult[i]) + ' val=' + str(validResult[i]) + '\n'
        
    MAE /= dataLen
    MSE /= dataLen

    # print and write result
    print('\n **** validation result ****')
    print('MAE=' + str(MAE))
    print('MSE=' + str(MSE))

    f = open('result_valid.txt', 'w')
    f.write(result + '\nMAE = ' + str(MAE) + ', MSE = ' + str(MSE))
    f.close()
