# 1. deepLearning_main.py #
input: DATA + INPUT AND OUTPUT INFO (COMPATIBLE) + MODEL -> output: TRAIN AND TEST RESULT

Execute Deep Learning and write the result of learning using given information.

## How to execute ##
```python deepLearning_main.py```

## input_output_info.txt: input and output data ##
* 1st line: (train input file name) (train input columns)
* 2st line: (train output file name) (train output columns)
* 3rd line: (test input file name) (test input columns)
* 4th line: (test output file name) (test output column options)

for each column X (train input/output and test input columns):
* ```X```: considered as NUMERIC values
* ```Xt```: considered as TEXT
* ```Xd```: considered as DATE/TIME (yyyy-mm-dd form)
* ```Xd1```: considered as DATE/TIME (mm/dd/yyyy form)
* ```Xl```: log2(NUMERIC value)'s
* ```Xz```: Z-value(NUMERIC value)'s
* ```Xlz```: Z-value(log2(NUMERIC value))'s
* ```Xlp```: log2(NUMERIC value + 1)'s
* ```Xlpz```: Z-value(log2(NUMERIC value + 1))'s

for each column X (test output columns):
* ```o```: output original data (ex: 2.3456)
* ```r```: output rounded data (ex: 2)

output file name:
(test output file name)

### example ###
```
input_example.csv 0 1z 2l 3lz 5t 6t 9d
output_example.csv 0 1 2lz 5t 7t 8d
test_example.csv 0 1z 2l 3lz 5t 6t 9d
test_result.csv r r o o o o
```

* line 1: name of train input data file is ```input_example.csv``` and in this file,
  * 0th~3th column is NUMERIC(1st column with Z-value, 2nd column with log-value, and 3rd column with Z-value(log-value))
  * 5th and 6th column is considered as TEXT
  * 9th column is DATE/TIME
* line 2: name of train output data file is ```output_example.csv``` and in this file,
  * 0th~2nd column is NUMERIC(2nd column with Z-value(log-value))
  * 3rd and 5th column is considered as TEXT
  * and 6th column is DATE/TIME
* line 3: name of test input data file is ```test_example.csv``` and in this file,
  * 0th~3rd column is NUMERIC(1st column with Z-value, 2nd column with log-value, and 3rd column with Z-value(log-value))
  * 5th and 6th column is considered as TEXT
  * 9th column is DATE/TIME
* line 4: name of test input data file is ```test_result.csv``` and in this file,
  * 0th~1st column output is rounded to integer
  * 2nd~5th column output is not rounded (original)

## deepLearning_model.txt: specify deep learning model ##
### layer settings ###
* Flatten (input): ```FI``` (tf.keras.layers.Flatten(input_shape=(len(trainI[0]),)))
* Flatten: ```F``` (keras.layers.Flatten())
* Dense: ```D X AF``` (keras.layers.Dense(X, activation=AF))
* Dense (output): ```DO AF``` (keras.layers.Dense(len(trainO[0]), activation=AF))
* Dropout: ```Drop X``` (keras.layers.Dropout(X))
* 2D convolution for input: ```C2DI X A B C D AF``` (keras.layers.Conv2D(X, kernel_size=(A, B), input_shape=(C, D, 1), activation=AF))
* 2D convolution: ```C2D X A B AF``` (keras.layers.Conv2D(X, (A, B), activation=AF))
* 2D max pooling:  ```MP X``` (keras.layers.MaxPooling2D(pool_size=X))
* Reshape: ```R X Y``` (tf.keras.layers.Reshape((X, Y, 1), input_shape=(X*Y,))

example:
* ```FI``` (tf.keras.layers.Flatten(input_shape=(len(trainI[0]),)))
* ```F``` (keras.layers.Flatten())
* ```D 16 relu``` (keras.layers.Dense(16, activation='relu'))
* ```DO sigmoid``` (keras.layers.Dense(len(trainO[0]), activation='sigmoid'))
* ```Drop 0.25``` (keras.layers.Dropout(0.25))
* ```C2DI 32 3 3 12 12 relu``` (keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(12, 12, 1), activation='relu'))
* ```C2D 32 3 3 relu``` (keras.layers.Conv2D(32, (3, 3), activation='relu'))
* ```MP 2``` (keras.layers.MaxPooling2D(pool_size=2))
* ```R 12 12``` (tf.keras.layers.Reshape((12, 12, 1), input_shape=(12*12,))

IMPORTANT: LAST LAYER should use SIGMOID as activation function.

### optimizer settings ###
* Adadelta: ```OP adadelta lr X eps``` (tf.keras.optimizers.Adadelta(learning_rate=lr, rho=X, epsilon=eps))
* Adagrad: ```OP adagrad lr X eps``` (tf.keras.optimizers.Adagrad(learning_rate=lr, initial_accumulator_value=X, epsilon=eps))
* Adam only with learning rate: ```OP adam0 lr``` (tf.keras.optimizers.Adam(lr))
* Adam: ```OP adam1 lr X Y eps``` (tf.keras.optimizers.Adam(learning_rate=lr, beta_1=X, beta_2=Y, epsilon=eps, amsgrad=False))
* Adamax: ```OP adamax lr X Y eps``` (tf.keras.optimizers.Adamax(learning_rate=lr, beta_1=X, beta_2=Y, epsilon=eps))
* Nadam: ```OP nadam lr X Y eps``` (tf.keras.optimizers.Nadam(learning_rate=lr, beta_1=X, beta_2=Y, epsilon=eps))
* RMSprop: ```OP rmsprop lr X A eps``` (tf.keras.optimizers.RMSprop(learning_rate=lr, rho=X, momentum=A, epsilon=eps))
* SGD: ```OP sgd lr X``` (tf.keras.optimizers.SGD(learning_rate=lr, momentum=X, nesterov=False))

example:
* ```OP adadelta 0.001 0.95 1e-07``` (tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07))
* ```OP adagrad 0.001 0.1 1e-07``` (tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07))
* ```OP adam0 0.001``` (tf.keras.optimizers.Adam(0.001))
* ```OP adam1 0.001 0.9 0.999 1e-07``` (tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False))
* ```OP adamax 0.001 0.9 0.999 1e-07``` (tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07))
* ```OP nadam 0.001 0.9 0.999 1e-07``` (tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07))
* ```OP rmsprop 0.001 0.9 0.0 1e-07``` (tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07))
* ```OP sgd 0.01 0.0``` (tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False))

### example ###
```
FI
D 1000 sigmoid
D 1000 sigmoid
DO sigmoid
OP adam0 0.001
```

* line 1: Flatten input layer
* line 2: Dense layer with 1000 nodes, using SIGMOID as activation function
* line 3: Dense layer with 1000 nodes, using SIGMOID as activation function
* line 4: Dense output layer, using SIGMOID as activation function
* line 5: Use ADAM optimizer with learning rate = 0.001 

# 2. image_data_generator.py #
input: IMAGE + INFO AND RULES -> output: DATA

Write data about train/test input/output images.
* including label using rules (```keywordX, valueX``` pairs in ```input_output_image_info.txt```)
* resize image for input of the neural network

## How to execute ##
```python image_data_generator.py```

## input_output_image_info.txt: information about images and how to write the data ##
in the form of
```
width height RW GW BW trainImagesFolder testImagesFolder
keyword0 value0
keyword1 value1
...
keywordN valueN
```
* width: width of image after resizing (for input of neural network)
* height: height of image after resizing (for input of neural network)
* RW, GW and BW: weight for Red, Green and Blue components of image
* trainImagesFolder: the name of folder containing train images
* testImagesFolder: the name of folder containing test images
* ```keywordX, valueX``` pairs: label it as ```valueX``` if the file name contains ```keywordX```

for example,
```
12 12 1 1 1 train_imgs test_imgs
naver 0
google 1
```
* width and height of image after resizing is ```12``` and ```12```. (images are converted info 12x12 images)
* RW, GW and BW are ```1```, ```1``` and ```1```. (weight of Red, Green and Blue are all the same)
* trainImagesFolder is ```train_imgs```. (image for training is in ```train_imgs``` folder)
* testImagesFolder is ```test_imgs```. (image for testing is in ```test_imgs``` folder)
* ```naver 0``` means, if the image file name contains ```naver```, it is labeled as ```0```.
* ```google 1``` means, if the image file name contains ```google```, it is labeled as ```1```.

# 3. input_output_info_converter.py #
input: INPUT AND OUTPUT INFO (NOT COMPATIBLE) -> INPUT AND OUTPUT INFO (COMPATIBLE)

Convert ranged input/output information into input/output information compatible with ```input_output_info.txt``` and ```deepLearning_main.py```. Write converted information into ```input_output_info_converted.txt```.

## How to execute ##
```python input_output_info_converter.py```

## input_output_info_original.txt: ranged information ##
for training input/output and test input
* ```start~end``` -> ```start start+1 ... end```
* ```start(label)~end(label)``` -> ```start(label) start+1(label) ... end(label)```

for example,
* ```0~9``` -> ```0 1 2 3 4 5 6 7 8 9```
* ```10~14t``` -> ```10t 11t 12t 13t 14t```
* ```15~19z``` -> ```15z 16z 17z 18z 19z```
* ```20~29d1``` -> ```20d1 21d1 22d1 23d1 24d1 25d1 26d1 27d1 28d1 29d1```

for test output
* ```o(number)``` -> ```o o o ... o```(```(number)``` o's)
* ```r(number)``` -> ```r r r ... r```(```(number)``` r's)

for example,
* ```o2``` -> ```o o```
* ```r3``` -> ```r r r```
* ```o5``` -> ```o o o o o```

# 4. naivebayes.py #
input: DATA + INPUT AND OUTPUT INFO (in ```input_output_info_nb.txt```) -> output: TRAIN AND TEST RESULT

Execute Naive Bayes algorithm and write the result of learning using given information.

## How to execute ##
```python naivebayes.py```

## input_output_info_nb.txt: input/output information for naivebayes.py ##
in the form of
```
*trainInputFileName
col0 rule0
col1 rule1
...
colX ruleX
*trainOutputFileName
col0 rule0
col1 rule1
...
colY ruleY
*testInputFileName
col0 rule0
col1 rule1
...
colZ ruleZ
*testOutputFileName
```
* ```trainInputFileName```: name of train input data file
* ```trainOutputFileName```: name of train output data file
* ```testInputFileName```: name of test input data file
* ```testOutputFileName```: name of test output data file
* ```colK ruleK```: pair of column no and categorization rule -> apply categorization rule ```ruleK``` to column ```colK``` of this file / the rules are as following:
  * ```None_```: do not apply categorization rule
  * ```cond0 label0 cond1 label1 ... condN labelN labelM```:
    * if ```cond0``` is TRUE then label it as ```label0```
    * else if ```cond1``` is TRUE then label it as ```label1```
    * ...
    * else if ```condN``` is TRUE then label it as ```labelN```
    * if all conditions are FALSE then label it as ```labelM```
  * about condition ```cond0```, ```cond1```, ..., and ```condN```
    * ```X```: equal to X
    * ```>X```, ```>=X```, ```<X```, ```<=X```, ```X```, ```<>X```: greater than/at least/less than/at most/equal to/not equal to X
    * ```*X*```, ```*X```, ```X*```: contains/end with/start with X
    * ```!*X*```, ```!*X```, ```!X*```: do not contains/end with/start with X

for example,
```
*input_example_nb.csv
0 None_
1 <10 a <20 b <30 c d
2 <10 x <20 y <30 z w
*input_example_nb.csv
4 None_
*test_example_nb.csv
1 None_
4 <10 a <20 b <30 c d
5 <10 x <20 y <30 z w
*test_example_output_nb.csv
```
* ```*input_example_nb.csv``` -> using ```input_example_nb.csv``` as input data file
* ```0 None_``` -> apply no categorization rule for 0-th column of input data file
* ```1 <10 a <20 b <30 c d``` -> apply categorization rule for 1st column of input data file, as same as the following:
  * if the value is less than 10(```<10```), label it as ```a```
  * else if the value is less than 20(```<20```), label it as ```b```
  * else if the value is less than 30(```<30```), label it as ```c```
  * else label it as ```d```
* ```2 <10 x <20 y <30 z w``` -> apply categorization rule for 2nd column of input data file, as same as the following:
  * if the value is less than 10(```<10```), label it as ```a```
  * else if the value is less than 20(```<20```), label it as ```b```
  * else if the value is less than 30(```<30```), label it as ```c```
  * else label it as ```d```
* ```*input_example_nb.csv``` -> using ```input_example_nb.csv``` as output data file
* ```4 None_``` -> apply no categorization rule for 4th column of output data file
* ```*test_example_nb.csv``` -> using ```test_example_nb.csv``` as test data file
* ```1 None_``` -> apply no categorization rule for 1st column of test data file
* ```4 <10 a <20 b <30 c d``` -> apply categorization rule for 4th column of test data file, as same as the following:
  * if the value is less than 10(```<10```), label it as ```a```
  * else if the value is less than 20(```<20```), label it as ```b```
  * else if the value is less than 30(```<30```), label it as ```c```
  * else label it as ```d```
* ```5 <10 x <20 y <30 z w``` -> apply categorization rule for 5th column of test data file, as same as the following:
  * if the value is less than 10(```<10```), label it as ```a```
  * else if the value is less than 20(```<20```), label it as ```b```
  * else if the value is less than 30(```<30```), label it as ```c```
  * else label it as ```d```
* ```*test_example_output_nb.csv``` -> write the test output data to ```test_example_output_nb.csv``` file
