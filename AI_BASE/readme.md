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
 * ```trainInput``` : train input data file
 * ```trainOutput``` : train output data file
 * ```testInput``` : test input data file
 * ```testOutput``` : test output data file (to create)
 * ```testOutputReal``` : real(actual) output for test input (compare it with ```testOutput``` to evaluate)
 * ```test_report``` : report for the test
 * ```valid_rate``` : ```0``` for training-test, or larger than ```0``` for validation
 * ```valid_report``` : report for the validation
 * ```modelConfig``` : refer to below
 * ```normalizeName``` : name of the file contains average and stddev
 * ```validInterval``` : interval for data to be set as validation data
   * for example, if this value is ```5```, validation rate is ```0.25``` and there are ```100``` training data, ```0~4, 14~19, 20~24, 55~59, 84~89```-th data can be set as validation data

### ```modelConfig``` file (for deep learning)
EXAMPLE - Refer to ```modelConfig.txt``` and ```modelConfig_example.txt```.

<strong>for model structure</strong>
```
I 60                        (keras.layers.InputLayer(input_shape=(60,)))
I1 60                       (keras.layers.InputLayer(input_shape=(60, 1)))
FI                          (keras.layers.Flatten(input_shape=(len(trainI[0]),)))
F                           (keras.layers.Flatten())
D 16 relu                   (keras.layers.Dense(16, activation='relu'))
DO sigmoid                  (keras.layers.Dense(len(trainO[0]), activation='sigmoid'))
Drop 0.25                   (keras.layers.Dropout(0.25))
C1DI 32 3 60 relu           (keras.layers.Conv1D(filters=32, kernel_size=3, input_shape=(60, 1), activation='relu'))
C1DI 32 3 60 relu same      (keras.layers.Conv1D(filters=32, kernel_size=3, input_shape=(60, 1), activation='relu', padding='same'))
C1D 32 3 relu               (keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
C1D 32 3 relu same          (keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
MP1D 2                      (keras.layers.MaxPooling1D(pool_size=2))
C2DI 32 3 3 12 12 relu      (keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(12, 12, 1), activation='relu'))
C2DI 32 3 3 12 12 relu same (keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(12, 12, 1), activation='relu', padding='same'))
C2D 32 3 3 relu             (keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
C2D 32 3 3 relu same        (keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
MP2D 2                      (keras.layers.MaxPooling2D(pool_size=2))
R1 12                       (keras.layers.Reshape((12, 1), input_shape=(12,))
R2 12 12                    (keras.layers.Reshape((12, 12, 1), input_shape=(12*12,))
BN                          (keras.layers.BatchNormalization())
```

<strong>for optimizer and loss</strong>
```
OP adadelta 0.001 0.95 1e-07    (tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07))
OP adagrad 0.001 0.1 1e-07      (tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07))
OP adam0 0.001                  (tf.keras.optimizers.Adam(0.001))
OP adam1 0.001 0.9 0.999 1e-07  (tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False))
OP adamax 0.001 0.9 0.999 1e-07 (tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07))
OP nadam 0.001 0.9 0.999 1e-07  (tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07))
OP rmsprop 0.001 0.9 0.0 1e-07  (tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07))
OP sgd 0.01 0.0                 (tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False))
LOSS mean_absolute_error        (set loss as mean_absolute_error)
```

Loss can be omitted from the ```modelConfig``` file.

Examples of loss: refer to https://keras.io/api/optimizers/

## 1. CODE FILE INFO
From line 44 to 59 of ```main.py```, you can see:
```
import decisionTree as _DT
import frequentWord as _FW
import kNN as _KNN
import makeDF as _DF
import makePCA as _PCA
import printData as _PD
import xgBoost as _XB
import textVec as _TV
import compareValid as _CV

# deep Learning
import deepLearning_GPU as DL
import deepLearning_GPU_helper as helper
import deepLearning_main as DLmain
import AIBASE_main as AImain
```

### methods (files for deep learning include some utilities)
<strong>FOR KNN (METHOD 0)</strong>
 * ```kNN.py``` : for using kNN algorithm
 
<strong>FOR DECISION TREE (METHOD 1)</strong>
 * ```DecisionTree.py```
   * ```createDTfromDF``` : create decision tree using dataFrame
   * ```predictDT``` : predict using Decision Tree
   
<strong>FOR TEXT VECTORIZATION (METHOD 2)</strong>
 * ```textVec.py```
   * ```penn2morphy```, ```lemmatize_sent``` and ```preprocess_text```
 * ```frequentWord.py```
   * UNUSED
   
<strong>FOR XGBOOST (METHOD 3 and 4)</strong>
 * ```xgBoost.py``` : for using xgBoost algorithm 
 
<strong>FOR DEEP LEARNING (METHOD 5 and 6) - include some utilities</strong>
 * ```AIBASE_main.py``` : read ```config.txt``` and execute deep learning using ```deepLearning_main.py```
 * ```deepLearning_GPU.py``` :
   * ```create_model``` : create a neural network for deep learning
   * ```learning``` : train neural network using inputs and outputs
   * ```deepLearning``` : train neural network
   * ```deepLearningModel``` : load the model from file
   * ```modelOutput``` : return the model output for test input
 * ```deepLearning_GPU_helper.py``` :
   * ```flatten``` : flatten 2d array to 1d array
   * ```sigmoid``` : sigmoid(x)
   * ```argmax``` : return max index of array
   * ```invSigmoid``` : inverse of sigmoid(x)
   * ```exceed``` : check whether the index exceeds feasible array range
   * ```arrayCopyFlatten``` : copy a part of 2d array and flatten to 1d array
   * ```arrayCopy``` : copy 2d array
   * ```roundedArray``` : print rounded array (to n decimals)
   * ```getDataFromFile``` : extract and return data array from file (delimiter is ```\t```)
   * ```getNN``` : return neural network according to model info
   * ```getOptimizer``` : return optimizer according to model info
 * ```deepLearning_main.py``` :
   * ```saveArray``` : save 2d array as file
   * ```denormalize``` : denormalize training/test output of deep learning
   * ```writeNormalizeInfo``` : write average and std-dev of train output array
   * ```writeTestResult``` : write test result file
   * ```deepLearning``` : deep learning process
   * ```dataFromDF``` : save train input, train output and test input as file, from dataFrame

### utilities
 * ```compareValid.py``` : compare validation prediction with validation data, and return MAE and MSE
 * ```makeDF.py```
   * ```getIndex``` : get index value of the column
   * ```makeDataFrame``` : create and return dataFrame from data file
 * ```makePCA.py``` : compute and return PCA from data file
 * ```printData.py``` : print data as set of data points on the 2d plane or 3d space
 * ```readData.py```:
   * ```extractColumns``` : extract columns from array
   * ```extractExceptForColumns```: extract data from array, except for columns
   * ```splitArray``` : split an array with multiple arrays, with conditions
   * ```mergeTestResult``` : merge test result arrays into one array
   * ```readPGN```: (for only Finding ELO from Kaggle)
   * ```saveArray```: save array as file, with specified splitter
   * ```loadArray```: load array from file, with specified splitter
   * ```getLogVal```: (for only Finding ELO from Kaggle)
   * ```textColumnToOneHot```: convert text column into one-hot columns
   
## 2. EXECUTION
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

### ```AIBASE_main.py``` file
Write ```config.txt``` as above before execution. 
