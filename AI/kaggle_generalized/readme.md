## How to execute ##
```python deepLearning_main.py```

## input_output_info.txt: input and output data ##
* 1st line: (train input file name) (train input columns)
* 2st line: (train output file name) (train output columns)
* 3rd line: (test input file name) (test input columns)

for each column X:
* ```X```: considered as NUMERIC values
* ```Xt```: considered as TEXT
* ```Xd```: considered as DATE/TIME

### example ###
```
input_example.csv 0 1 3t 4t 7d
output_example.csv 0 3t 5t 6d
test_example.csv 0 1 3t 4t 7d
```

* line 1: 0th and 1st column is NUMERIC, 3rd and 4th column is considered as TEXT, and 7th column is DATE/TIME.
* line 2: 0th column is NUMERIC, 3rd and 5th column is considered as TEXT, and 6th column is DATE/TIME.
* line 3: 0th and 1st column is NUMERIC, 3rd and 4th column is considered as TEXT, and 7th column is DATE/TIME.

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
* SGD: ```OP sgd 0.01 0.0``` (tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False))

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
* line 2: Dense layer with 1000 nodes
* line 3: Dense layer with 1000 nodes
* line 4: Dense output layer
* line 5: Use ADAM optimizer with learning rate = 0.001 
