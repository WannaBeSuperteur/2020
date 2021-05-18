import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np

if __name__ == '__main__':

    train_input_fn = 'train_input.txt'
    train_output_fn = 'train_output.txt'
    augment_test = False

    # read training input and output
    train_input = RD.loadArray(train_input_fn)
    train_output = RD.loadArray(train_output_fn)
    train_rows = len(train_input)

    print(' ==== before augmentation ====')
    print(np.shape(train_input))
    print(np.array(train_input))
    print('')
    print(np.shape(train_output))
    print(np.array(train_output))

    # augment training input and output
    for i in range(train_rows):
        train_input.append(train_input[i][::-1])

        if augment_test == True:
            train_output.append(train_output[i][::-1])
        else:
            train_output.append(train_output[i])

    print('\n ==== after augmentation ====')
    print(np.shape(train_input))
    print(np.array(train_input))
    print('')
    print(np.shape(train_output))
    print(np.array(train_output))

    # save
    RD.saveArray('train_input_augmented.txt', train_input, '\t', 1)
    RD.saveArray('train_output_augmented.txt', train_output, '\t', 1)
