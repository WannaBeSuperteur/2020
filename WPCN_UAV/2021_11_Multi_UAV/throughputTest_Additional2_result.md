## execution
execution: ```throughputTest_Additional2.py```

## training and test settings
training iteration : 0 ~ 474 (475 iterations)
test iteration : 0 ~ 24 (25 iterations)

## algorithm
for each UAV and every time slot,
 * with probability ```p```, UAV goes to the direction of the mean of NOT communicated devices
   * If UAV did NOT communicated with device ```0``` at ```(20, 4)```, device ```1``` at ```(14, 8)```, the direction is toward the mean ```(17, 6)```.
   * When the UAV is at ```(10, 6)```, it moves to positive direction of ```x``` to go to ```(17, 6)```.
 * with probability ```1-p```, UAV goes to the INVERSE direction of the mean of communicated devices
   * If UAV communicated with device ```2``` at ```(5, 10)```, device ```3``` at ```(3, 8)```, the direction is toward the mean ```(4, 9)```.
   * When the UAV is at ```(2, 15)```, it moves to positive direction of ```x``` and negative direction of ```y``` to go to ```(4, 9)```.
 * Find the best possible case of movement to move to the desired location.

## result
<b>worse than genetic-based algorithm</b>, have to try another algorithm (for example, just go toward the NEAREST NOT-COMMUNICATED device)

the set of ```{model prediction,ground truth}``` (in ```train_valid_result.csv```):
(the prediction converged to average)

```
0,0.002478353213518858,0.0
1,0.002478353213518858,0.0
2,0.002478353213518858,0.0
3,0.002478353213518858,0.0
4,0.002478353213518858,0.0
5,0.002478353213518858,0.0
6,0.002478353213518858,0.0
7,0.002478353213518858,0.0592683937746396
8,0.002478353213518858,0.0
9,0.002478353213518858,0.0
10,0.002478353213518858,0.0
11,0.002478353213518858,0.00489502979538
12,0.002478353213518858,0.0
13,0.002478353213518858,0.0
14,0.002478353213518858,0.0067027431205183
15,0.002478353213518858,0.0
16,0.002478353213518858,0.0042856860365533
17,0.002478353213518858,0.0
18,0.002478353213518858,0.0
19,0.002478353213518858,0.0081974721812855
```

 * To check the trajectory, run ```throughputTest_Additional2.py```.
