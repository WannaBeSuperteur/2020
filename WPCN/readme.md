## NOW Version 1 (200511)
https://github.com/WannaBeSuperteur/2020/tree/91df3ce2e1f9fcce97bea12fb67323d81daa6b20 (deleted)

### Experiment Result
* refer to ```report/200519_WPCN.pptx```

### Changes of main code
* ```WPCN_helper_REAL.py```
  * use learning rate 5.0 -> 1.0e+10
  * use interval 1.0e-9 instead of 1.0 to find a differential coefficient for chargeTimeList
* ```WPCN_paper.py```
  * delete the mode using integer value of uStar[0] and uStar[1], so only using exact value for uStar
  * use learning rate 5000.0 -> 3.0e+08
  * use iteration times 800 -> 7000
  * use interval 1.0e-6 instead of 1.0 to find a differential coefficient for minOfTgivenV1
* ```deepLearning_WPCN_REAL_GPU_xxxxxx.py```
  * remove rounding of optiY_test and optiX_test
  * add algorithm to modify optiY_test and optiX_test
    * optiY_test = (down - up)/(up + this + down)
    * optiX_test = (right - left)/(left + this + right)

### Python code
#### main
* for training and testing of each algorithm (Sum- or Common- throughput maximization) using throughput map
  * create the throughput map if throughput map file (optiInfoForMap txt file) does not exist
  * ```deepLearning_WPCN_REAL_GPU_200318.py``` : Sigmoid [0, 1]
  * ```deepLearning_WPCN_REAL_GPU_200318_.py``` : Sigmoid [0, 1]
  * ```deepLearning_WPCN_REAL_GPU_200326.py``` : Sigmoid [-1, 1]
  * ```deepLearning_WPCN_REAL_GPU_200330.py``` : Sigmoid [-1, 1] using (2x^3-9)/7 from x = 1 to 2
  * ```deepLearning_WPCN_REAL_GPU_200401.py``` : Sigmoid [-2, 2]
  * ```deepLearning_WPCN_REAL_GPU_200403.py``` : Tanh [-1, 1]
  * ```deepLearning_WPCN_REAL_GPU_200409.py``` : Sigmoid [-1.5, 1.5]
  * ```deepLearning_WPCN_REAL_GPU_200412.py``` : Tanh [-0.75, 0.75]
* helper file (get throughput for given wireless device position list)
  * ```WPCN_helper_REAL.py```

#### experiments
* using average location of all wireless devices
  * ```WPCN_avgLocation.py```
* using fixed location (from y=0,x=0 to y=11,x=11)
  * ```WPCN_fixedLocation.py```
* implementation of paper for number of HAPs = 1 (Placement Optimization of Energy and Information Access Points in Wireless Powered Communication Networks) and test using the implementation
  * ```WPCN_paper.py```
  
#### compare throughput map (in ```/optiInfoForMap```)
* ```optiInfoForMap/compare.py```

### Text files and images
#### training and testing maps
* ```DL_WPCN_xxxx.txt``` (created when running ```deepLearning_WPCN_REAL_GPU_xxxxxx.py``` with value 1 for ```0->read files, 1->create new files``` option)

#### metadata for training and testing
* ```map.txt```

#### throughput maps
* throughput map for Sum throughput maximization
  * ```optiInfoForMap/optiInfoForMap_0_ver1.txt```
* throughput map for Common throughput maximization
  * ```optiInfoForMap/optiInfoForMap_1_ver1.txt```

#### result files
* result of experiment (```deepLearning_WPCN_REAL_GPU_xxxxxx.py```)
  * Sum throughput maximization
    * all files whose name is in the form of ```results/DL_WPCN_xx_result_0.txt```
  * Common throughput maximization
    * all files whose name is in the form of ```results/DL_WPCN_xx_result_1.txt```

#### result for experiments
* using average location of all wireless devices
  * Common throughput maximization
    * ```results_experiment/DL_WPCN_result_1_avg_ver200511.txt```
    * ```results_experiment/DL_WPCN_result_1_avg_ver200511_print.txt```
* using fixed location (from y=0,x=0 to y=11,x=11)
  * Common throughput maximization
    * ```results_experiment/DL_WPCN_result_1_fixed_ver200511.txt```
    * ```results_experiment/DL_WPCN_result_1_fixed_ver200511_print.txt```
* throughput and HAP optimization test
  * ```image/helper_ver_May00_common_test.png```
  
### Report files
* Excel Report
  * ```report/200511_testResult.xlsx```
  
* Visualized Throughput Map
  * ```report/200511_visualList.xlsx```

## Version 0 (200318)
https://github.com/WannaBeSuperteur/2020/tree/e515244d41bc7d12caa8f0020e2821211e6745ad/WPCN (deleted)

### Experiment Result
* refer to ```report/200429_WPCN.pptx```

### Python code
#### main
* for training and testing of each algorithm (Sum- or Common- throughput maximization) using throughput map
  * create the throughput map if throughput map file (optiInfoForMap txt file) does not exist
  * ```deepLearning_WPCN_REAL_GPU_200318.py``` : Sigmoid [0, 1]
  * ```deepLearning_WPCN_REAL_GPU_200318_.py``` : Sigmoid [0, 1]
  * ```deepLearning_WPCN_REAL_GPU_200326.py``` : Sigmoid [-1, 1]
  * ```deepLearning_WPCN_REAL_GPU_200330.py``` : Sigmoid [-1, 1] using (2x^3-9)/7 from x = 1 to 2
  * ```deepLearning_WPCN_REAL_GPU_200401.py``` : Sigmoid [-2, 2]
  * ```deepLearning_WPCN_REAL_GPU_200403.py``` : Tanh [-1, 1]
  * ```deepLearning_WPCN_REAL_GPU_200409.py``` : Sigmoid [-1.5, 1.5]
  * ```deepLearning_WPCN_REAL_GPU_200412.py``` : Tanh [-0.75, 0.75]
* helper file (get throughput for given wireless device position list)
  * ```WPCN_helper_REAL.py```

#### experiments
* using fixed location (from y=0,x=0 to y=11,x=11)
  * ```WPCN_fixedLocation.py```
* implementation of paper for number of HAPs = 1 (Placement Optimization of Energy and Information Access Points in Wireless Powered Communication Networks) and test using the implementation
  * ```WPCN_paper.py```

### Text files and images
#### training and testing maps
* ```DL_WPCN_xxxx.txt``` (created when running ```deepLearning_WPCN_REAL_GPU_200326.py``` with value 1 for ```0->read files, 1->create new files``` option)

#### metadata for training and testing
* ```map.txt```

#### throughput maps
* throughput map for Sum throughput maximization
  * ```optiInfoForMap/optiInfoForMap_0_ver0.txt```
* throughput map for Common throughput maximization
  * ```optiInfoForMap/optiInfoForMap_1_ver0.txt```

#### result files
* result of paper (```WPCN_paper.py```)
  * Sum throughput maximization
    * ```results_paper/DL_WPCN_result_0_paper_ver0.txt```
    * ```results_paper/DL_WPCN_result_0_paper_ver1.txt```
    * ```results_paper/DL_WPCN_result_0_paper_ver1_print.txt```
  * Common throughput maximization
    * ```results_paper/DL_WPCN_result_1_paper_ver0.txt```
    * ```results_paper/DL_WPCN_result_1_paper_ver0_print.txt```
    * ```results_paper/DL_WPCN_result_1_paper_ver1.txt```
    * ```results_paper/DL_WPCN_result_1_paper_ver1_print.txt```
    * ```results_paper/DL_WPCN_result_1_paper_ver2.txt```
    * ```results_paper/DL_WPCN_result_1_paper_ver2_print.txt```
    * ```results_paper/DL_WPCN_result_1_paper_ver3.txt```
    * ```results_paper/DL_WPCN_result_1_paper_ver3_print.txt```
    
* result of experiment (```deepLearning_WPCN_REAL_GPU_xxxxxx.py```)
  * Sum throughput maximization
    * all files whose name is in the form of ```results/DL_WPCN_result_0_xxxxxxxxxx.txt```
  * Common throughput maximization
    * all files whose name is in the form of ```results/DL_WPCN_result_1_xxxxxxxxxx.txt```

#### result for experiments
* using fixed location (from y=0,x=0 to y=11,x=11)
  * Sum throughput maximization
    * ```results_experiment/DL_WPCN_result_0_fixed.txt```
  * Common throughput maximization
    * ```results_experiment/DL_WPCN_result_1_fixed.txt```
    * ```results_experiment/DL_WPCN_result_1_fixed_print.txt```
* throughput and HAP optimization test
  * ```image/helper_ver_Mar00_common_test.png```

### Report files
* PPT files
  * result of ```deepLearning_WPCN_REAL_GPU_200318.py``` and ```deepLearning_WPCN_REAL_GPU_200318_.py```
    * ```report/200324_WPCN.pptx```
    * ```report/200421_WPCN.pptx```
  * result of ```deepLearning_WPCN_REAL_GPU_200326.py``` and ```deepLearning_WPCN_REAL_GPU_200330.py```
    * ```report/200331_WPCN.pptx```
    * ```report/200421_WPCN.pptx```
  * result of ```deepLearning_WPCN_REAL_GPU_200401.py``` and ```deepLearning_WPCN_REAL_GPU_200403.py```
    * ```report/200407_WPCN.pptx```
    * ```report/200421_WPCN.pptx```
  * result of ```deepLearning_WPCN_REAL_GPU_200409.py``` and ```deepLearning_WPCN_REAL_GPU_200412.py```
    * ```report/200414_WPCN.pptx```
    * ```report/200429_WPCN.pptx```

* Excel Report
  * ```report/200318_testResult.xlsx```
  
* Visualized Throughput Map
  * ```report/200318_visualList.xlsx```

## Unused files
* ```Qlearning_deep_WPCN.py```
* ```WPCN_helper.py```
* ```WPCN_helper_renewal.py```
* ```deepLearning_WPCN.py```
* ```deepLearning_WPCN_REAL_GPU.py```
* ```deepLearning_WPCN_REAL_GPU_200305.py```
* ```deepLearning_WPCN_REAL_GPU_200306.py```
* ```deepLearning_WPCN_REAL_GPU_200311.py```
* ```deepLearning_WPCN_renewal.py```
* ```deepLearning_WPCN_renewal_GPU.py```
