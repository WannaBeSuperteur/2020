## algo_base
Using base code ```algorithm_base.py```, run our own algorithm code to solve this WPCN-UAV problem.

## base settings
The input of our deep learning model contains ```(2 * windowSize + 1) * (2 * windowSize + 1) + PARAMS``` input features and ```1``` output feature.
* ```(2 * windowSize + 1) * (2 * windowSize + 1)``` input features for CNN (Convolutional) model
* ```PARAMS``` input features for parameters

The settings are described in ```base_settings.txt```.
* ```paramCells``` : the number of parameters to use (max ```4```)
* ```p0_cases``` : the number of possible cases (```0```, ..., ```1```) for parameter ```p0``` of the input of the deep learning model
* ```p1_cases``` : the number of possible cases for parameter ```p1```
* ```p2_cases``` : the number of possible cases for parameter ```p2```
* ```p3_cases``` : the number of possible cases for parameter ```p3```

Our **trained** model finds the **best combination** of {```p0```, ```p1```, ```p2``` and ```p3```} which maximizes the ```output``` value of our model, given the ```(2 * windowSize + 1) * (2 * windowSize + 1)``` parameters.

## base functions
There are three ```base function```s.
* ```base_func_initializeMovementOfUAV(devices)``` : return the visit order of devices in the cluster
  * for example: ```[4, 0, 2, 3, 1, 5]```
* ```computeDirectionList(bestParams, q, l, N, deviceListC, initialMovement, width, height)``` : return the direction list ```directionList```.
  * for example: ```[2, 5, 8, 8, 0, 7, 8, 8, 8, 1, 1]```
    * ```0, 1, 2, ..., 7``` means moving 5m with the direction of ```0, 45, 90, ..., 315``` degree
    * ```8``` means stop
* ```getDeviceLocation(l, w, final_throughputs)``` : return the location of the device to visit at next time (probably is null)
  * in the form of ```(x, y)``` which means ```x``` and ```y``` location of the device

## usage
import :

```import algorithm_base as algo_base```

basic usage of base settings : refer to ```base_settings.txt```
```
paramCells=2 # the number of parameter cells in the input of deep learning model
p0_cases=11 # the number of possible cases for parameter 0
p1_cases=11 # the number of possible cases for parameter 1
p2_cases=1 # the number of possible cases for parameter 2
p3_cases=1 # the number of possible cases for parameter 3
```

basic usage of base functions :
```
algo_base.throughputTest(M, T, N, L, ...,
                         base_func_initializeMovementOfUAV=func0,
                         base_func_computeDirectionList=func1,
                         base_func_getDeviceLocation=func2)
```
* ```func0```, ```func1```, ```func2``` are the three ```base function```s.

refer to examples:
* ```throughputTest_NewGenetic.py``` (refactorized on ```Jul 05, 2022```)
