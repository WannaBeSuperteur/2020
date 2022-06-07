Minimum Throughput Maximization for Multi-UAV Enabled WPCN: A Deep Reinforcement Learning Method

https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8950047

Reference:

 * JIE TANG and JINGRU SONG et al, "Minimum Throughput Maximization for Multi-UAV Enabled WPCN: A Deep Reinforcement Learning Method", IEEE Access, 2020. Available online at https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8950047&tag=1.

## FILE INFO
### helper files
* ```algorithms.py```
  * ```kMeansClustering(L, deviceList, width, height, H, N, display, saveImg, verbose)``` : K means clustering (PHASE 2 of FIGURE 2)
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

### main files
* common
  * **functions :**
    * ```main``` : get configuration info using ```helper.py``` and write them to ```config_info.txt```
    * ```moveUAV(q, directionList, N, L, width, height)``` : move the UAV using an element of the direction list indexed by (```l = 0,1,...L-1``` and) ```t = 0,1,...,N-1```
    * ```changeColor(colorCode, k)``` : change the color based on the color given by the color code (```k``` is brightness)
    * ```DEEP_LEARNING_MODEL(tf.keras.Model)``` : **MAIN DEEP LEARNING MODEL** including convolutional and dense layers
    * ```defineAndTrainModel(train_input, train_output, test_input, test_output, epochs, windowSize)``` :
      * 1. define and configure time model
      * 2. train the model using ```train_input``` and ```train_output```
      * 3. save the model as ```WPCN_UAV_DL_model```
      * 4. test (predict the ```test_output``` as ```test_prediction```) and save the result as ```train_valid_result.csv```  
    * ```getAndTrainModel(epochs, windowSize)``` :
      * 1. load the preprocessed input and output data, ```train_input_preprocessed.csv``` and ```train_output_preprocessed.csv```
      * 2. convert the preprocessed data into numpy array
      * 3. divide the data into training/test data
      * 4. define and train model using ```defineAndTrainModel()``` function
    * ```saveTrajectoryGraph(iterationCount, width, height, w, all_throughputs, q, markerColors, training, isStatic)``` : plot the trajectory data ```w``` at iteration ```iterationCount```, and save the figure of the data as ```{static/train/test}_trajectory_iter_{iterationCount}.png```
  * **input :**
    * (```getAndTrainModel(...)```) ```train_input_preprocessed.csv``` and ```train_output_preprocessed.csv```, preprocessed training input and output data
  * **output :**
    * (```main```) ```config_info.txt```, the configuration info is written
    * (```defineAndTrainModel(...)```) ```WPCN_UAV_DL_model```, the model
    * (```defineAndTrainModel(...)```) ```train_valid_result.csv```, the result of training and validation
    * (```saveTrajectoryGraph(...)```) ```{static/train/test}_trajectory_iter_{iterationCount}.png```, the plotted trajectory data at iteration ```iterationCount```
* ```throughputTest.py```
* ```throughputTest_Genetic.py```
* ```throughputTest_Additional.py```
  * ```getIndexOfDirection(directionX, directionY)``` : get the index of the direction using ```np.arctan(directionY / directionX)```, and decide ```xy_change``` (index of the change of ```x``` and ```y```) using it

### train and test many times at once
* ```throughputTest_at_once.py```
  * do many ```train-and-test``` iterations at once, using manual settings of ```iters```, ```L```, ```devices``` and ```Ns```(array)
  * **output :** minimum throughput list at ```iters```, ```L```, ```devices``` and ```N```(in ```Ns```)
    * in ```txt``` form : ```minThroughputList_iter_{iters}_L_{L}_devs_{devs}_N_{N}.txt```
    * in ```csv``` form : ```minThroughputList_iter_{iters}_L_{L}_devs_{devs}_N_{N}.csv```
