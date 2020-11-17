## 0. BASIC INFO
### You should install below before running:
 * Python 3.7.x
 * h5py 2.10.0
 * Keras 2.3.0
 * Keras-Applications 1.0.8
 * Keras-Preprocessing 1.1.2
 * numpy 1.18.1
 * tensorflow 2.0.0
 * tensorflow-estimator 2.0.1
 * tensorflow-gpu 2.0.0a0
 * tensorflow-gpu-estimator 2.1.0

### usage in Python code (3.7.x)
This is an example.
```
import sys
sys.path.insert(0, '../../AI_BASE')
import deepLearning_GPU as DL
import deepLearning_GPU_helper as helper
import deepLearning_main as DLmain
import AIBASE_main as main
```

### config.txt (for deep learning)
EXAMPLE - Refer to ```config.txt```.
```
trainInput TR_I.txt
trainOutput TR_O.txt
testInput TE_I.txt
testOutput TE_O.txt
testOutputReal TE_OR.txt
test_report TE_Report.txt
valid_rate 0
valid_report VA_Report.txt
modelConfig modelConfig.txt
normalizeName dddd.txt
```

### ```modelConfig``` file (for deep learning)
EXAMPLE - Refer to ```modelConfig.txt``` and ```modelConfig_example.txt```.
```
I 60                   (keras.layers.InputLayer(input_shape=(60,)))
I1 60                  (keras.layers.InputLayer(input_shape=(60, 1)))
FI                     (keras.layers.Flatten(input_shape=(len(trainI[0]),)))
F                      (keras.layers.Flatten())
D 16 relu              (keras.layers.Dense(16, activation='relu'))
DO sigmoid             (keras.layers.Dense(len(trainO[0]), activation='sigmoid'))
Drop 0.25              (keras.layers.Dropout(0.25))
C1DI 32 3 60 relu      (keras.layers.Conv1D(filters=32, kernel_size=3, input_shape=(60, 1), activation='relu'))
C1D 32 3 relu          (keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
MP1D 2                 (keras.layers.MaxPooling1D(pool_size=2))
C2DI 32 3 3 12 12 relu (keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(12, 12, 1), activation='relu'))
C2D 32 3 3 relu        (keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
MP2D 2                 (keras.layers.MaxPooling2D(pool_size=2))
R1 12                  (keras.layers.Reshape((12, 1), input_shape=(12,))
R2 12 12               (keras.layers.Reshape((12, 12, 1), input_shape=(12*12,))
```

## 1. EXECUTION
### ```main.py``` file
You can use the function below:
```
executeAlgorithm(finalResult, trainName, testName, ftype, fcolsTrain, fcolsTest, targetColName, exceptCols, # basic configuration
                     globalValidationRate, method, usePCA, validationExceptCols, validationCol, # important configuration
                     PCAdimen, exceptTargetForPCA, useLog, logConstant, # PCA and log
                     kNN_k, kNN_useAverage, kNN_useCaseWeight, kNN_weight, # for kNN (method 0)
                     DT_maxDepth, DT_criterion, DT_splitter, DT_numericRange, # for Decision Tree (method 1)
                     exceptColsForMethod2, # for method 2
                     XG_info, # xgBoost (for method 3 and 4)
                     DL_normalizeTarget) # for Deep Learning (method 5 and 6)
```
If you want some example, refer to line 413~502 of ```main.py```. (some of files do not exist)

<strong>basic configuration (cannot be omitted)</strong>
 * ```finalResult``` : init as ```None```
 * ```trainName``` : training data file
 * ```testName``` : test data file
 * ```ftype``` : ```json```, ```csv``` or ```txt```
 * ```fColsTrain``` : list of columns in the training data file
 * ```fColsTest``` : list of columns in the test data file
 * ```targetColName``` : target column for machine learning
 * ```exceptCols``` : do not contain these columns in training/valid/test data
 * ```globalValidationRate``` : validation rate for method ```0```, ```1```, ```2```, ```3``` and ```4```
   * If larger than ```0```, then split training data into training data and validation data, according to the value.
   * For deep learning (method ```5``` or ```6```), validation rate is ```valid_rate``` in the file ```config.txt```.
 * ```method``` :
   * ```0``` for PCA (optional) + kNN (k Nearest Neighbors)
   * ```1``` for PCA (optional) + DT (Decision Tree)
   * ```2``` for TextVec (Text Vectorization) + NB (Naive Bayes) - there could be some errors
   * ```3``` for PCA + XgBoost - there could be some errors
   * ```4``` for XgBoost only - there could be some errors
   * ```5``` for PCA + deep learning
   * ```6``` for deep learning only
 * ```usePCA``` : use PCA? (```True``` or ```False```, for method ```0``` and ```1```)
 * ```validationExceptCols``` : do not use these columns for validation data (from training data file)
 * ```validationCol``` : column to validate for validation (for training data file)
   * index of ```targetColName```

<strong>PCA and log settings (cannot be omitted)</strong>
 * ```PCAdimen``` : dimension for PCA
 * ```exceptTargetForPCA``` : do not include target column for computing PCA (```True``` or ```False```)
 * ```useLog``` : using log(x) for numeric data columns (```True``` or ```False```)
 * ```logConstant``` : transform ```x``` into ```log2(x + logConstant)```
 
<strong>method settings (can be omitted for non-corresponding methods)</strong>
 * method ```0``` : kNN
   * ```kNN_k``` : value of ```k```
   * ```kNN_useAverage``` : use average value of neighbors (```True``` or ```False```)
   * ```kNN_useCaseWeight``` : use weight by (1 / the number of cases) (```True``` or ```False```)
   * ```kNN_weight``` : weight for each test input data column
 * method ```1``` : Decision Tree
   * ```DT_maxDepth``` : maximum depth of decision tree
   * ```DT_criterion``` : ```gini``` or ```entropy```
   * ```DT_splitter``` : ```best``` or ```random```
   * ```DT_numericRange``` : convert to mean of each range when target column is numeric
     * For example, when it is ```[10, 30, 50, 70]``` and target column values are ```[3, 5, 15, 25, 35, 45, 55, 65, 75, 77]```, then it is converted to ```[3, 5, 20, 20, 40, 40, 60, 60, 75, 77]```.
 * method ```2``` : TextVec + NB
   * ```exceptColsForMethod2```
 * method ```3``` and ```4``` : xgBoost
   * ```XG_info``` : ```[XG_epochs, XG_boostRound, XG_earlyStoppingRounds, XG_foldCount, XG_rateOf1s]```, where
     * ```XG_epochs``` : xgboost training epochs
     * ```XG_boostRound``` : num_boost_round for xgb.train
     * ```XG_earlyStoppingRounds``` : early_stopping_rounds for xgb.train
     * ```XG_foldCount``` : if the value is ```N```, then (N-1)/N of whole dataset is for training, 1/N is for test
     * ```XG_rateOf1s``` : rate of 1's (using this, portion of 1 should be XG_rateOf1s)
 * method ```5``` and ```6``` : deep learning
   * ```DL_normalizeTarget``` : normalize training output value? (```False``` because automatically normalized)
