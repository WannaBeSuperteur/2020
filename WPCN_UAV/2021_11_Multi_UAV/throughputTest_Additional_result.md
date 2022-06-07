## execution
execution: ```throughputTest_Additional.py```

## training and test settings
training iteration : 0 ~ 499 (500 iterations)
test iteration : 0 ~ 24 (25 iterations)

## algorithm
for each UAV and every time slot,
 * with probability ```probChooseMinThroughput```, UAV goes to the direction of the device with CURRENTLY MINIMUM throughput
 * with probability ```1.0 - probChooseMinThroughput```, UAV goes to the direction of the RANDOMLY SELECTED device
 * Find the best possible case of movement to move to the desired location.

## result
<b>better than genetic-based algorithm</b>, with more dataset and improvement of output value metric (```tanh(x)``` -> ```tanh(10x)```), correlation coefficient is around ```+0.771```

the set of ```{model prediction,ground truth}``` (in ```train_valid_result.csv```):
(the prediction converged to average)

```
0,0.08914025872945786,0.1748862081718538
1,0.2839721143245697,0.291437633927996
2,0.470866858959198,0.4418779270048413
3,0.041988223791122437,0.0
4,0.07595589011907578,0.053867499341611
5,0.4357355833053589,0.7210051871199148
6,0.0734139233827591,0.0563123337093411
7,0.268530935049057,0.2491082911605382
8,0.12132488936185837,0.1067694296844266
9,0.2386953979730606,0.2088424826318562
10,0.3752231001853943,0.3336532450356291
11,0.07232648134231567,0.0
12,0.26189717650413513,0.0796085268589592
13,0.41238829493522644,0.2401165616551357
14,0.060347698628902435,0.0
15,0.3508436977863312,0.2894421645528793
16,0.08493777364492416,0.0
17,0.23950575292110443,0.3956607110555948
18,0.32169926166534424,0.2881134225481518
19,0.08159013837575912,0.1766762825277861
20,0.08172319084405899,0.1553984663353708
21,0.22157853841781616,0.1112447904862941
22,0.3142123818397522,0.3725932039780938
23,0.12432555109262466,0.1379085575424556
24,0.1792113333940506,0.3976117166319531
```

 * To check the trajectory, run ```throughputTest_Additional.py```.