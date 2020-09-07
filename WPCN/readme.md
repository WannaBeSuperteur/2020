## NOW for final paper (200714)

* THERE WAS A CRITICAL ISSUE FOR THE PAPER. PLEASE REFER TO https://github.com/WannaBeSuperteur/2020/commit/83665628f374752bc467a9d4e7f6edce74462c6a.

### Experiment Guidelines
* For our methodology (CT.RATE (%) for our methodology): refer to ```guideline_ourMethod.txt```
* For the methodology in the original paper (CT.RATE (%) for the methodology in the original paper): refer to ```guideline_original.txt```
* For CT.AVGMAX: refer to ```optiInfoForMap/compare_result_200615.txt``` (run ```optiInfoForMap/compare.py``` and input as the file)
* Example experiment result file: ```exampleTestResult.zip```

### Paper
* ```paper01_temporary.docx``` <strong>(updated for the issue)</strong>

### Experiment Result
* refer to ```../report/200714_WPCN.pptx``` <strong>(updated for the issue)</strong>

### Python code
#### main
* <strong>MAIN EXPERIMENT:</strong> for training and testing (Sum- or Common- throughput maximization) using throughput map
  * create the throughput map if throughput map file (optiInfoForMap txt file) does not exist
  * ```deepLearning_WPCN_forPaper.py``` : Sigmoid [-1, 1]
* helper file (get throughput for given wireless device position list)
  * ```WPCN_helper_REAL_paper.py```

#### experiments
* <strong>ORIGINAL PAPER EXPERIMENT:</strong> implementation of the original paper for number of HAPs = 1 (Placement Optimization of Energy and Information Access Points in Wireless Powered Communication Networks) and test using the implementation
  * ```WPCN_paper_forPaper.py``` <strong>(updated for the issue)</strong>
  
#### compare throughput map (in ```/optiInfoForMap```)
* ```optiInfoForMap/compare.py```

### Text files and images
#### training and testing maps
* ```originalMaps_size_WDs/DL_WPCN_xxxx.txt``` (created when running ```deepLearning_WPCN_forPaper.py``` with value 1 for ```0->read files, 1->create new files``` option)
  * size: ```8```, ```12``` or ```16```
  * WDs: ```6``` or ```10```

#### metadata for training and testing
* ```map.txt``` (each line denotes each configuration)

#### throughput maps
* throughput map for Sum throughput maximization
  * ```optiInfoForMap/optiInfoForMap_0_forPaper_8_6.txt``` (size=8, WDs=6)
  * ```optiInfoForMap/optiInfoForMap_0_forPaper_8_10.txt``` (size=8, WDs=10)
  * ```optiInfoForMap/optiInfoForMap_0_forPaper_12_6.txt``` (size=12, WDs=6)
  * ```optiInfoForMap/optiInfoForMap_0_forPaper_12_10.txt``` (size=12, WDs=10)
  * ```optiInfoForMap/optiInfoForMap_0_forPaper_16_6.txt``` (size=16, WDs=6)
  * ```optiInfoForMap/optiInfoForMap_0_forPaper_16_10.txt``` (size=16, WDs=10)
* throughput map for Common throughput maximization
  * ```optiInfoForMap/optiInfoForMap_1_forPaper_8_6.txt``` (size=8, WDs=6)
  * ```optiInfoForMap/optiInfoForMap_1_forPaper_8_10.txt``` (size=8, WDs=10)
  * ```optiInfoForMap/optiInfoForMap_1_forPaper_12_6.txt``` (size=12, WDs=6)
  * ```optiInfoForMap/optiInfoForMap_1_forPaper_12_10.txt``` (size=12, WDs=10)
  * ```optiInfoForMap/optiInfoForMap_1_forPaper_16_6.txt``` (size=16, WDs=6)
  * ```optiInfoForMap/optiInfoForMap_1_forPaper_16_10.txt``` (size=16, WDs=10)

#### result files
* <strong>We only calculated CT.RATE (%) of each method and CT.AVGMAX. CT.AVERAGE is derived by CT.RATE * CT.AVGMAX for each method, and PR is derived by (CT.AVERAGE of our method) / (CT.AVERAGE of the methodology in the original paper).</strong>
* <strong>result of experiment</strong> (```deepLearning_WPCN_forPaper.py```)
  * Sum throughput maximization
    * all files whose name is in the form of ```results/DL_WPCN_forPaper_0_(size)_(WDs).txt```
  * Common throughput maximization
    * <strong>for CT.RATE (%) for our methodology in the paper</strong>
    * all files whose name is in the form of ```results/DL_WPCN_forPaper_1_(size)_(WDs).txt```
  * size: ```8```, ```12``` or ```16```
  * WDs: ```6``` or ```10```
* <strong>result of the original paper</strong> (```WPCN_paper_forPaper.py```) <strong>(updated for the issue)</strong>
  * Sum throughput maximization <strong>(updated for the issue)</strong>
    * all files whose name is in the form of ```results_paper/DL_WPCN_result_0_paper_forPaper_(size)_(WDs).txt```
    * all files whose name is in the form of ```results_paper/DL_WPCN_result_0_paper_forPaper_(size)_(WDs)_print.txt```
  * Common throughput maximization <strong>(updated for the issue)</strong>
    * <strong>for CT.RATE (%) for the methodology in the original paper</strong>
    * all files whose name is in the form of ```results_paper/DL_WPCN_result_1_paper_forPaper_(size)_(WDs).txt```
    * all files whose name is in the form of ```results_paper/DL_WPCN_result_1_paper_forPaper_(size)_(WDs)_print.txt```
  * size: ```8```, ```12``` or ```16```
  * WDs: ```6``` or ```10```
* result of paper with 2 options, ```K``` and ```et``` (```WPCN_paper_forPaper_option.py```) <strong>(updated for the issue)</strong>
  * Common throughput maximization
    * all files whose name is in the form of ```results_paper_option/DL_WPCN_result_1_paper_forPaper_(size)_(WDs)_K=(K)_et=(et).txt```
    * all files whose name is in the form of ```results_paper_option/DL_WPCN_result_1_paper_forPaper_(size)_(WDs)_K=(K)_et=(et)_print.txt```
  * size: ```8```, ```12``` or ```16```
  * WDs: ```6``` or ```10```
  * K: ```10```, ```20``` or ```40```
  * et: ```1.0e-05```, ```1.0e-06``` or ```1.0e-07```
* compare result of throughput
  * <strong>for CT.AVGMAX in the paper (common for each method)</strong>
  * ```optiInfoForMap/compare_result_200615.txt```
  * ```sum train```: sum of throughput for 900 training maps
  * ```sum test```: sum of throughput for 100 test maps
  * ```sum all```: sum of throughput for 1000 training/test maps
  * ```avg train```: average of throughput for 900 training maps
  * ```avg test```: average of throughput for 100 test maps
  * ```avg all```: average of throughput for 1000 training/test maps
  * ```<<< problem 0 >>>```: SUM-THROUGHPUT maximization
  * ```<<< problem 1 >>>```: COMMON-THROUGHPUT maximization
  
### Report files
* Visualized Throughput Map
  * ```../report/200615_visualList_8_6.xlsx``` (size=8, WDs=6)
  * ```../report/200615_visualList_8_10.xlsx``` (size=8, WDs=10)
  * ```../report/200615_visualList_12_6.xlsx``` (size=12, WDs=6)
  * ```../report/200615_visualList_12_10.xlsx``` (size=12, WDs=10)
  * ```../report/200615_visualList_16_6.xlsx``` (size=16, WDs=6)
  * ```../report/200615_visualList_16_10.xlsx``` (size=16, WDs=10)

## NOW Version 2 (200519)
https://github.com/WannaBeSuperteur/2020/tree/ac16a8ca221159d3294949f56b0a166845ece998

### Experiment Result
* refer to ```../report/200526_WPCN.pptx```

### Changes of main code
* ```WPCN_helper_REAL.py```
  * read each map file in WPCN/originalMaps (https://github.com/WannaBeSuperteur/2020/commit/bdeeb8814bef7408f5e46fddd6750972e136304c)
  * "different lr and interval for Sum- and Common-" (https://github.com/WannaBeSuperteur/2020/commit/23f1f541446074652d50f3a8b5f8511652f68821)
* ```WPCN_paper.py```
  * read each map file in WPCN/originalMaps (https://github.com/WannaBeSuperteur/2020/commit/bdeeb8814bef7408f5e46fddd6750972e136304c)
  * use new text file name except maps and config (https://github.com/WannaBeSuperteur/2020/commit/a5aeab39238b238a4ef8a2ca9d75819ca176cb96)
* ```deepLearning_WPCN_REAL_GPU_xxxxxx.py```
  * read each map file in WPCN/originalMaps (https://github.com/WannaBeSuperteur/2020/commit/bdeeb8814bef7408f5e46fddd6750972e136304c)
  * use new text file name except maps and config (https://github.com/WannaBeSuperteur/2020/commit/a5aeab39238b238a4ef8a2ca9d75819ca176cb96)
  * revise optiInfo-making algorithm (https://github.com/WannaBeSuperteur/2020/commit/ed0c6d7aedf840dcbaa96d13a35c8d7e39d41bdb)

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
* ```originalMaps/DL_WPCN_xxxx.txt``` (created when running ```deepLearning_WPCN_REAL_GPU_xxxxxx.py``` with value 1 for ```0->read files, 1->create new files``` option)

#### metadata for training and testing
* ```map.txt```

#### throughput maps
* throughput map for Sum throughput maximization
  * ```optiInfoForMap/optiInfoForMap_0_200519.txt``` (same as ```optiInfoForMap/optiInfoForMap_0_ver1.txt```)
* throughput map for Common throughput maximization
  * ```optiInfoForMap/optiInfoForMap_1_200519.txt``` (same as ```optiInfoForMap/optiInfoForMap_1_ver0.txt```)

#### result files
* result of experiment (```deepLearning_WPCN_REAL_GPU_xxxxxx.py```)
  * Sum throughput maximization
    * all files whose name is in the form of ```results/DL_WPCN_ver200519_xx_result_0.txt```
  * Common throughput maximization
    * all files whose name is in the form of ```results/DL_WPCN_ver200519_xx_result_1.txt```
* result of paper (```WPCN_paper.py```)
  * Sum throughput maximization
    * ```results_paper/DL_WPCN_result_0_paper_ver200519.txt```
    * ```results_paper/DL_WPCN_result_0_paper_ver200519_print.txt```
  * Common throughput maximization
    * ```results_paper/DL_WPCN_result_1_paper_ver200519.txt```
    * ```results_paper/DL_WPCN_result_1_paper_ver200519_print.txt```

#### result for experiments
* using average location of all wireless devices (created by ```WPCN_avgLocation.py```)
  * Sum throughput maximization
    * ```results_experiment/DL_WPCN_result_0_avg_ver200519.txt```
    * ```results_experiment/DL_WPCN_result_0_avg_ver200519_print.txt```
  * Common throughput maximization
    * ```results_experiment/DL_WPCN_result_1_avg_ver200519.txt```
    * ```results_experiment/DL_WPCN_result_1_avg_ver200519_print.txt```
* using fixed location (from y=0,x=0 to y=11,x=11) (created by ```WPCN_fixedLocation.py```)
  * Sum throughput maximization
    * ```results_experiment/DL_WPCN_result_0_fixed_ver200519.txt```
    * ```results_experiment/DL_WPCN_result_0_fixed_ver200519_print.txt```
  * Common throughput maximization
    * ```results_experiment/DL_WPCN_result_1_fixed_ver200519.txt``` (same as ```results_experiment/DL_WPCN_result_1_fixed.txt```)
    * ```results_experiment/DL_WPCN_result_1_fixed_ver200519_print.txt```
* throughput and HAP optimization test (when running ```WPCN_helper_REAL.py```)
  * Sum throughput maximization
    * ```image/helper_ver_May01_sum_test.png```
  * Common throughput maximization
    * ```image/helper_ver_May01_common_test.png```
  
### Report files
* Excel Report
  * ```../report/200519_testResult.xlsx```
  
* Visualized Throughput Map
  * ```../report/200519_visualList.xlsx```
    * Sum throughput maximization: same as ```../report/200511_visualList.xlsx``` (ver1)
    * Common throughput maximization: same as ```../report/200318_visualList.xlsx``` (ver0)

## Version 1 (200511)
https://github.com/WannaBeSuperteur/2020/tree/dffe6a35686f4cb17c49f348af0966137a953ef1

### Experiment Result
* refer to ```../report/200519_WPCN.pptx```

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
* using average location of all wireless devices (created by ```WPCN_avgLocation.py```)
  * Common throughput maximization
    * ```results_experiment/DL_WPCN_result_1_avg_ver200511.txt```
    * ```results_experiment/DL_WPCN_result_1_avg_ver200511_print.txt```
* using fixed location (from y=0,x=0 to y=11,x=11) (created by ```WPCN_fixedLocation.py```)
  * Common throughput maximization
    * ```results_experiment/DL_WPCN_result_1_fixed_ver200511.txt```
    * ```results_experiment/DL_WPCN_result_1_fixed_ver200511_print.txt```
* throughput and HAP optimization test (when running ```WPCN_helper_REAL.py```)
  * ```image/helper_ver_May00_common_test.png```
  
### Report files
* Excel Report
  * ```../report/200511_testResult.xlsx```
  
* Visualized Throughput Map
  * ```../report/200511_visualList.xlsx```

## Version 0 (200318)
https://github.com/WannaBeSuperteur/2020/tree/e515244d41bc7d12caa8f0020e2821211e6745ad/WPCN (deleted)

### Experiment Result
* refer to ```../report/200429_WPCN.pptx```

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
* using fixed location (from y=0,x=0 to y=11,x=11) (created by ```WPCN_fixedLocation.py```)
  * Sum throughput maximization
    * ```results_experiment/DL_WPCN_result_0_fixed.txt```
  * Common throughput maximization
    * ```results_experiment/DL_WPCN_result_1_fixed.txt```
    * ```results_experiment/DL_WPCN_result_1_fixed_print.txt```
* throughput and HAP optimization test (when running ```WPCN_helper_REAL.py```)
  * ```image/helper_ver_Mar00_common_test.png```

### Report files
* PPT files
  * result of ```deepLearning_WPCN_REAL_GPU_200318.py``` and ```deepLearning_WPCN_REAL_GPU_200318_.py```
    * ```../report/200324_WPCN.pptx```
    * ```../report/200421_WPCN.pptx```
  * result of ```deepLearning_WPCN_REAL_GPU_200326.py``` and ```deepLearning_WPCN_REAL_GPU_200330.py```
    * ```../report/200331_WPCN.pptx```
    * ```../report/200421_WPCN.pptx```
  * result of ```deepLearning_WPCN_REAL_GPU_200401.py``` and ```deepLearning_WPCN_REAL_GPU_200403.py```
    * ```../report/200407_WPCN.pptx```
    * ```../report/200421_WPCN.pptx```
  * result of ```deepLearning_WPCN_REAL_GPU_200409.py``` and ```deepLearning_WPCN_REAL_GPU_200412.py```
    * ```../report/200414_WPCN.pptx```
    * ```../report/200429_WPCN.pptx```

* Excel Report
  * ```../report/200318_testResult.xlsx```
  
* Visualized Throughput Map
  * ```../report/200318_visualList.xlsx```

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
* ```originalMaps/optiInfoForMap_0_forPaper_8_6.txt```
* ```originalMaps/optiInfoForMap_0_forPaper_8_10.txt```
* ```originalMaps/optiInfoForMap_0_forPaper_12_6.txt```
* ```originalMaps/optiInfoForMap_0_forPaper_12_10.txt```
* ```originalMaps/optiInfoForMap_0_forPaper_16_6.txt```
* ```originalMaps/optiInfoForMap_0_forPaper_16_10.txt```
* ```originalMaps/optiInfoForMap_1_forPaper_8_6.txt```
* ```originalMaps/optiInfoForMap_1_forPaper_8_10.txt```
* ```results_paper/DL_WPCN_forPaper_0_8_6.txt```
* ```results_paper/DL_WPCN_forPaper_0_8_10.txt```
* ```results_paper/DL_WPCN_forPaper_0_12_6.txt```
* ```results_paper/DL_WPCN_forPaper_0_12_10.txt```
* ```results_paper/DL_WPCN_forPaper_0_16_6.txt```
* ```results_paper/DL_WPCN_forPaper_0_16_10.txt```
* ```results_paper/DL_WPCN_forPaper_1_8_6.txt```
* ```results_paper/DL_WPCN_forPaper_1_8_10.txt```
