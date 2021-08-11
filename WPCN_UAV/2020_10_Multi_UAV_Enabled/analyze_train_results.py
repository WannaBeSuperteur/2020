import numpy as np
import pandas as pd

if __name__ == '__main__':

    # train_loss for last epoch, last 3 epochs, and last 5 epochs
    # val_loss   for last epoch, last 3 epochs, and last 5 epochs
    # train_acc  for last epoch, last 3 epochs, and last 5 epochs
    # val_acc    for last epoch, last 3 epochs, and last 5 epochs
    analyze_result = []
    iteration = 0
    episodes = 40

    for e in range(episodes):

        # load *.csv file about episode e
        episode_data = pd.read_csv('train_result_iter_' + str(iteration) + '_episode_' + str(e) + '.csv')
        epochs = episode_data.shape[0]

        print(e)

        # train loss
        train_loss = episode_data['train_loss']
        train_loss_last = train_loss.iloc[epochs-1]
        train_loss_last3 = train_loss.iloc[epochs-3:].mean()

        if epochs < 5:
            train_loss_last5 = train_loss.mean()
        else:
            train_loss_last5 = train_loss.iloc[epochs-5:].mean()

        # validation loss
        val_loss = episode_data['val_loss']
        val_loss_last = val_loss.iloc[epochs-1]
        val_loss_last3 = val_loss.iloc[epochs-3:].mean()

        if epochs < 5:
            val_loss_last5 = val_loss.mean()
        else:
            val_loss_last5 = val_loss.iloc[epochs-5:].mean()

        # train accuracy
        train_acc = episode_data['train_acc']
        train_acc_last = train_acc.iloc[epochs-1]
        train_acc_last3 = train_acc.iloc[epochs-3:].mean()

        if epochs < 5:
            train_acc_last5 = train_acc.mean()
        else:
            train_acc_last5 = train_acc.iloc[epochs-5:].mean()

        # validation accuracy
        val_acc = episode_data['val_acc']
        val_acc_last = val_acc.iloc[epochs-1]
        val_acc_last3 = val_acc.iloc[epochs-3:].mean()

        if epochs < 5:
            val_acc_last5 = val_acc.mean()
        else:
            val_acc_last5 = val_acc.iloc[epochs-5:].mean()

        # append to analyze result
        analyze_result.append([train_loss_last, train_loss_last3, train_loss_last5,
                               val_loss_last, val_loss_last3, val_loss_last5,
                               train_acc_last, train_acc_last3, train_acc_last5,
                               val_acc_last, val_acc_last3, val_acc_last5])

    # save the analyze result
    analyze_result = pd.DataFrame(analyze_result)
    analyze_result.columns = ['train_loss_last', 'train_loss_last3', 'train_loss_last5',
                              'val_loss_last', 'val_loss_last3', 'val_loss_last5',
                              'train_acc_last', 'train_acc_last3', 'train_acc_last5',
                              'val_acc_last', 'val_acc_last3', 'val_acc_last5']
    analyze_result.to_csv('train_analyze_result.csv')
