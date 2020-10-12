## You should install below before running:
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

## usage in Python code (3.7.x)
```
import sys
sys.path.insert(0, '../../AI_BASE')
import deepLearning_GPU as DL
import deepLearning_GPU_helper as helper
import deepLearning_main as DLmain
import AIBASE_main as main
```

## config.txt
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
```

## ```modelConfig``` file
EXAMPLE - Refer to ```modelConfig.txt``` and ```modelConfig_example.txt```.
```
I 60                   (keras.layers.InputLayer(input_shape=(60,)))
FI                     (keras.layers.Flatten(input_shape=(len(trainI[0]),)))
F                      (keras.layers.Flatten())
D 16 relu              (keras.layers.Dense(16, activation='relu'))
DO sigmoid             (keras.layers.Dense(len(trainO[0]), activation='sigmoid'))
Drop 0.25              (keras.layers.Dropout(0.25))
C2DI 32 3 3 12 12 relu (keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(12, 12, 1), activation='relu'))
C2D 32 3 3 relu        (keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
MP2D 2                 (keras.layers.MaxPooling2D(pool_size=2))
R1 12                  (keras.layers.Reshape((12, 1), input_shape=(12,))
R2 12 12               (keras.layers.Reshape((12, 12, 1), input_shape=(12*12,))
