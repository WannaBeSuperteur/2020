Source: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset#time_series_covid_19_confirmed.csv

## 1. Prepare raw data
> input: ```rawData_train.txt``` and ```rawData_test.txt```
* raw data for training and test
* Download CSV files from the source and create the files by copying the data.

### input example
* ```rawData_train.txt``` and ```rawData_test.txt```
```
30  40  45  50  51  55  56  57  57  60  62  65  70  80  100
```

## 2. run makeTrainData.py
> input: ```rawData_train.txt``` or ```rawData_test.txt```

> output: ```print_paper_train.txt```, ```print_paper_test.txt```, ```paper_train.txt```, ```paper_test.txt``` and ```info.txt```
* file name of raw data: ```rawData_train.txt```(train time) or ```rawData_test.txt```(test time).
* file name of train or test data: ```paper_train.txt```(train time) or ```paper_test.txt```(test time).
* Input atLeast, colsEachTrain, and after. (refer to instruction showed when running the code)

### output example (when atLeast=50, colsEachTrain=5 and after=7)
* ```print_paper_train.txt``` and ```print_paper_test.txt```
```
50  51  55  56  57  100
```
* ```paper_train.txt``` and ```paper_test.txt```
```
-0.9091  -0.7273  0  0.1818  0.3636  0.7544
```
* ```info.txt```
```
value value must be at least 50 (=C)
number of columns of each training data is 5 (=N)
train and test outputs are 7 days later of last input data (=K)
```

## 3. run main.py
> input: ```info.txt```, ```paper_train.txt``` and ```paper_test.txt```

> output: ```paper_result.txt``` and ```paper_table.txt```
* epoch=50 and scalingRate=4

### output example
* ```paper_result.txt```
```
 < epoch = 50 / scalingRate = 4 >
difference                 -> sum: 963.7447395242148 avg: 17.522631627712997
dif when test out is all 0 -> sum: 1128.3160192975072 avg: 20.51483671450013
accuracy rate              -> 1.170762311869571
True positive : 55 / 100.0%
True negative : 0 / 0.0%
False positive: 0 / 0.0%
False negative: 0 / 0.0%
positive      : 55 / 100.0%
negative      : 0 / 0.0%
correct       : 55 / 100.0%
```
* ```paper_table.txt```
```
epoch	scaling	#test	difAvg	0difAvg	accur	TP	TN	FP	FN	P	N	correct
note: difAvg error (squared) and 0-difAvg error (squared) are multiplied by 100
50	4	55	1752.2632	2051.4837	1.1708	100.0	0.0	0.0	0.0	100.0	0.0	100.0
```

## 4. calculate difference
* 0. Open ```test.xlsx```.
* 1. Copy output (in the form of 'test out: [A] val: [B] -> dif: [C]') of ```main.py``` and copy it to column A.
* 2. Extract all [A]s and copy them to column H, and change them to numeric value.
* 3. Extract all [B]s and copy them to column I, and change them to numeric value.
* 4. Open ```print_paper_test.txt``` and copy the column indicating the last day of input.
* 5. Compare last+test (prediction) and last+val (validation) values.
