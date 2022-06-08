Minimum Throughput Maximization for Multi-UAV Enabled WPCN: A Deep Reinforcement Learning Method

ORIGINAL BASELINE PAPER : https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8950047

Reference:

 * JIE TANG and JINGRU SONG et al, "Minimum Throughput Maximization for Multi-UAV Enabled WPCN: A Deep Reinforcement Learning Method", IEEE Access, 2020. Available online at https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8950047&tag=1.

## RUN INFO
command:
* ```python throughputTest.py```
or
* ```python throughputTest_Genetic.py```
or
* ```python throughputTest_Additional.py``` (latest)

to iterate many "train-test"s at once,
* ```python throughputTest_at_once.py```

## FILE INFO - HELPER
* ```algorithms.py```
  * ```kMeansClustering(L, deviceList, width, height, H, N, display, saveImg, verbose)``` : K means clustering **(PHASE 2 of FIGURE 2)**
    * plot the clustering result and save it as ```clustering_result.png```
  * **input :** None
  * **output :** ```clustering_result.png```, the device clustering result
* ```draw_log_graph.py```
  * 1. load the test result ```minThroughputList_iter_{iters}_L_{L}_devs_{devs}_N_{N}.txt``` files and read ```data_iters```, ```data_mean```, ```data_std``` and ```data_nonzero```.
  * 2. convert the data into numpy array
  * 3. plot the test result as picture
  * 4. save the plotted test result as ```minThroughputLog.png```
  * **input :** ```minThroughputList_iter_{iters}_L_{L}_devs_{devs}_N_{N}.txt```, the summary of test result of each iteration ```n```
  * **output :** ```minThroughputLog.png```, the summary of these test results
* ```formula.py```
  * the functions with formulas for WPCN-UAV research
* ```helper.py```
  * ```loadSettings(args)``` : load arguments from ```settings.txt```
  * ```printTime(funcName, inout)``` : log the execution time of functions, and write the log at ```time_log.txt```
  * **input :** ```settings.txt```, argument list
  * **output :** ```time_log.txt```, the log for the function execution time
* ```time_checker.py```
  * summarize execution time for each case specified as ```{func},{inout}``` with ```time```, ```count``` and ```avgTime```(average time)

## FILE INFO - MAIN
### main files - findBestParams function
* ```findBestParams(model, inputImage)``` of **```throughputTest_Additional.py```** : for each variable ```p0```, ```p1```, ```p2``` and ```p3``` with value range ```{0,1,2,3,4}```, **((D) and (E) of FIGURE 3, a part of PHASE 4 of FIGURE 2)**
  * 1. with ```params = [p0*0.25, p1*0.25, p2*0.25, p3*0.25]```, create input data as the concatenation of ```(inputImage, params)```
  * 2. with this input data, get the output ```outputOfModifiedParam``` of this input using ```model```
  * ```bestParams``` is the ```params``` with the best (largest) value of ```outputOfModifiedParam```
* ```findBestParams(model, inputImage, ranges)``` of **```throughputTest_Genetic.py```** : for each parameter in ```ranges```
  * ```ranges``` is in the form of ```[[a1, b1], [a2, b2], [a3, c3], ...]```, and ```[a1, b1]``` means parameter ```1``` with range ( -> can have value between) ```[a1, b1]```
  * 1. with initial ```params``` (initial ```bestParams``` = ```params```), create input data as the concatenation of ```(inputImage, params)```
  * 2. modify each parameter of ```bestParams``` slightly with ```+``` and ```-``` direction
    * only one selected parameter per modification, for example, ```[1.0, 2.5, 5.0]``` (```3``` parameters) -> ```[1.1, 2.5, 5.0], [0.9, 2.5, 5.0], [1.0, 2.25, 5.0], [1.0, 2.75, 5.0], [1.0, 2.5, 4.5], [1.0, 2.5, 5.5]``` (```6``` cases, ```2``` cases/parameter)
  * 3. create input data ```(inputImage, bestParams)``` using each modified parameter set in ```2.```
  * 4. with this input data, get the output ```outputOfModifiedParam``` of this input using ```model```
    * for model input, we convert ```bestParams``` properly
  * 5. if we find the largest ```outputOfModifiedParam``` value, then the corresponding parameters becomes ```bestParams``` in ```2.``` and go to ```2.```
  * 6. if we did not find it, return ```bestParams``` as NOT-CONVERTED-FOR-MODEL (original) version.
* ```findBestParams()``` of **```throughputTest.py```** : **DOES NOT EXIST**

### main files - throughputTest(...) function
#### file name : ```throughputTest_Additional.py```
* 1. create list of devices **(PHASE 1 of FIGURE 2)**
* 2. do clustering (if clustering failed, go to ```1.```) **(PHASE 2 of FIGURE 2)**
* 3. initialize and sort array ```alkl``` (```a_l,kl[n]``` of paper) as ```value```=```0```, the set of ```[i, k, j, t, value]```
  * ```i```, ```j``` : ```i = 0,1,...,L-1```, ```j = 0,1,...,L-1```
  * ```k```, ```t``` : ```k = (each device)```, ```t = 0,1,...,N-1```
  * ```value``` : the value of ```a_l,kl[n]```
* 4. for each UAV ```l``` in ```0,1,...,L-1``` (```L``` : the number of UAV),
  * 1. make input image using ```makeInputImage()``` function and flatten them
  * 2. find best parameter when ```training = False```, or create randomly decided parameter when ```training = True```
    * ```training = True``` (```training phase``` of our paper) : make randomly decided parameter set, and then train using the input and output derived using the parameter set
    * ```training = False``` (```test phase``` of our paper) : find the best parameter using the ```model``` and ```input image``` : **bestParams = findBestParams(model, inputImage)**
  * 3. initialize direction list ```directionList```, whose elements are for each time ```0,1,...,N-1```
  * 4. for each time slot ```t = 0,1,...,N-1```,
    * 1. move the UAV using ```moveUAV()``` function
    * 2. update array ```alkl``` (```a_l,kl[n]``` in the paper) using ```update_alkl()``` function
    * 3. get throughput using ```formula_11()``` function
    * 4. when ```isStatic``` (static mode) is false, decide the next movement of te UAV
      * find the direction to move UAV, using the nearest device derived by ```getDeviceLocation()``` function
  * 5. save throughput values if first iteration (```iterationCount = 0```)
    * throughput value file name: ```{static/train/test}_thrputs_iter_{iterationCount}_cluster_{l}_final.csv```
* 5. save input and output data
  * input data file name: ```{static/train/test}_input_raw.csv```
  * output data file name: ```{static/train/test}_output_raw.csv```
  * the number of columns and rows:
    * input data columns: ```(2 * windowSize + 1)^2 + 4```
    * output data columns: ```1```
    * input and output data rows: ```L * iterations```
* 6. preprocess the input and output data using ```preprocessInputAndOutput()``` function
* 7. save the preprocessed input and output data
  * preprocessed input data file name: ```{static/train/test}_input_preprocessed.csv```
  * preprocessed output data file name: ```{static/train/test}_output_preprocessed.csv```
* 8. create minimum/all throughput information, and save trajectory as graph using ```saveTrajectoryGraph()``` function

#### file name : ```throughputTest_Genetic.py```
difference from ```throughputTest_Additional.py``` is ...

1. **```4-3``` and ```4-4```** of **4. for each UAV ```l``` in ```0,1,...,L-1``` (```L``` : the number of UAV)**

* 4. for each UAV ```l``` in ```0,1,...,L-1``` (```L``` : the number of UAV),
  * 0. same as ```4-1``` and ```4-2``` of ```throughputTest_Additional.py``` (note: parameter -> within specific ranges)
  * 1. each parameter is called ```rounds```, ```pPercent```, and ```caseToFindInit```
  * 2. initialize direction list ```directionList```, whose size is ```3*3*3```
  * 3. for each ```round``` in ```rounds```,
    * 1. get throughput list of ```device```s, ```final_throughputs``` and find the minimum throughput ```minThroughput```=```min(final_throughputs)```
    * 2. define probability ```p``` using ```pPercent```, ```currentRound``` (the number indicating the current round) and ```rounds```
    * 3. define the number of cases to find ```casesToFind``` using ```casesToFindInit``` and ```currentRound```
    * 4. find the near cases (each case is ```nearCase```), and save info about each case
      * with probability ```p```, randomly modify movement
      * with probability ```1-p```, randomly swap two movements
    * 5. find the best (maximize common/minimum throughput) near case using the saved info, and update the direction list ```directionList``` of the bast case

2. **run ```5.```, ```6.``` and ```7.``` of ```throughputTest_Additional.py``` just after ```4-3.``` of above, so they become ```4-4.```, ```4-5.``` and ```4-6.``` respectively**

### main files - common
common (at least 2 of the files below)
* **functions :**
  * ```main``` : get configuration info using ```helper.py``` and write them to ```config_info.txt```
  * ```moveUAV(q, directionList, N, L, width, height)``` : move the UAV using an element of the direction list indexed by (```l = 0,1,...L-1``` and) ```t = 0,1,...,N-1``` **(PHASE 3 of FIGURE 2)**
  * ```changeColor(colorCode, k)``` : change the color based on the color given by the color code (```k``` is brightness)
  * ```DEEP_LEARNING_MODEL(tf.keras.Model)``` : **MAIN DEEP LEARNING MODEL** including convolutional and dense layers **(FIGURE 4)**
  * ```defineAndTrainModel(train_input, train_output, test_input, test_output, epochs, windowSize)``` :
    * 1. define and configure time model
    * 2. train the model using ```train_input``` and ```train_output```
    * 3. save the model as ```WPCN_UAV_DL_model```
    * 4. test (predict the ```test_output``` as ```test_prediction```) and save the result as ```train_valid_result.csv``` **(FIGURE 5)**
  * ```getAndTrainModel(epochs, windowSize)``` :
    * 1. load the preprocessed input and output data, ```train_input_preprocessed.csv``` and ```train_output_preprocessed.csv```
    * 2. convert the preprocessed data into numpy array
    * 3. divide the data into training/test data
    * 4. define and train model using ```defineAndTrainModel()``` function
  * ```saveTrajectoryGraph(iterationCount, width, height, w, all_throughputs, q, markerColors, training, isStatic)``` : plot the trajectory data ```w``` at iteration ```iterationCount```, and save the figure of the data as ```{static/train/test}_trajectory_iter_{iterationCount}.png```
  * ```update_alkl(alkl, q, w, l, t, N, s, b1, b2, mu1, mu2, fc, c, alphaP, numOfDevs, devices, isStatic)``` : update ```a_l,kl[n]``` in the paper = array ```alkl``` of the code file, and sort the array ```alkl```
  * ```preprocessInputAndOutput(input_data, output_data, windowSize)``` : for each row of ```input_data```,
    * 1. add ```(2 * windowSize + 1)``` x ```(2 * windowSize + 1)``` sized input data to flatten preprocessed input array ```preprocessed_input```.
    * 2. add ```bestParams``` with ```4``` variables as ORIGINAL value, to the same input array.
    * 3. Then the shape of flatten preprocessed input array becomes ```((2 * windowSize + 1)^2 + 4)```.
    * 4. for better performance of training, apply ```tanh(10 * output_data[i][0])``` into the flatten preprocessed output array ```preprocessed_output_data```.
  * ```makeInputImage(q, l, N, w, windowSize, sqDist)```: make the input image describing the current UAV and device location **((A), (B), and (C) of FIGURE 3, a part of PHASE 4 of FIGURE 2)**
  * ```getDeviceLocation(l, w, final_throughputs, probChooseMinThroughput)```: choose the device with minimum ```throughput```, or choose one of them randomly
  * ```saveMinThroughput(minThroughputList, memo)```: save minimum throughput list (based on ```minThroughputList```) as ```*.csv``` or ```*.txt``` file
    * mimimum throughput list file name: ```minThroughputList_{memo}_iter_{iters}_L_{L}_devs_{devs}_N_{N}.{csv/txt}```

### main files - functions exist only in specific file
* ```throughputTest.py``` only
  * ```convertToNumeric(value)``` : convert ```value``` to numeric (```int``` or ```float```)
  * ```modifyArr(arr, y, x, value, window)``` : clipping the value in the array between ```-1.5``` and ```1.5```
  * ```markDevicePosition(board, board_x, board_y, thrput, window)``` : mark the position of device in the ```board``` in the somewhat-Gaussian-like clipped form
  * ```getDevicePos(w, l): find the cluster (UAV) indexed by ```l``` from ```w```, and return the ```x``` and ```y``` position of the device in the cluster
  * ```getDistBetweenDeviceAndUAV(i, dev_x, dev_y, UAV_x, UAV_y)```: get the distance between the deivce and the UAV
  * ```makeBoard(thrput, w, l, window, width, height)```: mark device position of the devices in the cluster ```l```
  * ```computeOutput(q_current, q_after, thrput, thrput_after, w, l)```: compute the reward value
  * and so on
* ```throughputTest_Genetic.py``` only
  * ```getThroughput(...)``` : compute throughput value based on the direction list ```directionList```
* ```throughputTest_Additional.py``` only
  * ```getIndexOfDirection(directionX, directionY)``` : get the index of the direction using ```np.arctan(directionY / directionX)```, and decide ```xy_change``` (index of the change of ```x``` and ```y```) using it

### main files - input and output
* **input :**
  * (```getAndTrainModel(...)```) ```train_input_preprocessed.csv``` and ```train_output_preprocessed.csv```, preprocessed training input and output data
  * (```saveMinThroughput(...)```) ```minThroughputList_{memo}_iter_{iters}_L_{L}_devs_{devs}_N_{N}.{csv/txt}```, minimum throughput list
* **output :**
  * (```main```) ```config_info.txt```, the configuration info is written
  * (```defineAndTrainModel(...)```) ```WPCN_UAV_DL_model```, the model
  * (```defineAndTrainModel(...)```) ```train_valid_result.csv```, the result of training and validation
  * (```saveTrajectoryGraph(...)```) ```{static/train/test}_trajectory_iter{iterationCount}.png```, the plotted trajectory data at iteration ```iterationCount```
  * (```throughputTest(...)```) ```{static/train/test}_thrputs_iter_{iterationCount}_cluster_{l}_final.csv```
  * (```throughputTest(...)```) ```{static/train/test}_{input/output}_{raw/preprocessed}.csv```

## FILE INFO - TRAIN AND TEST MANY TIMES AT ONCE
* ```throughputTest_at_once.py```
  * do many ```train-and-test``` iterations at once, using manual settings of ```iters```, ```L```, ```devices``` and ```Ns```(array)
  * **output :** minimum throughput list at ```iters```, ```L```, ```devices``` and ```N```(in ```Ns```)
    * in ```txt``` form : ```minThroughputList_iter_{iters}_L_{L}_devs_{devs}_N_{N}.txt```
    * in ```csv``` form : ```minThroughputList_iter_{iters}_L_{L}_devs_{devs}_N_{N}.csv```
