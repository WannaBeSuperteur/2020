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
* ```throughputTest.py```
* ```throughputTest_Genetic.py```

### train and test many times at once
* ```throughputTest_at_once.py```
  * do many ```train-and-test``` iterations at once, using manual settings of ```iters```, ```L```, ```devices``` and ```Ns```(array)
  * **output :** minimum throughput list at ```iters```, ```L```, ```devices``` and ```N```(in ```Ns```)
    * in ```txt``` form : ```minThroughputList_iter_{iters}_L_{L}_devs_{devs}_N_{N}.txt```
    * in ```csv``` form : ```minThroughputList_iter_{iters}_L_{L}_devs_{devs}_N_{N}.csv```
