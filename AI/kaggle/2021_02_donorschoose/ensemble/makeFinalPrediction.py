# remove warings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

# compute and return ROC-AUC score
# ground_truth : label      , shape=(rows, 1)
# prediction   : probability, shape=(rows, numClass)
def computeROCAUC(prediction, ground_truth):

    prediction = np.array(prediction)
    ground_truth = np.array(ground_truth)

    return roc_auc_score(ground_truth, prediction)

# compute metric score between merged prediction with weights
def computeMetric(merged_predictions, weights, ground_truth):

    # compute merged prediction with the sum of (merged_prediction * weight)
    # shape of both merged_prediction and train_Y : (train_rows, numClass)
    for i in range(len(weights)):
        if i == 0:
            merged_prediction = weights[0] * merged_predictions[0]
        else:
            merged_prediction += weights[i] * merged_predictions[i]
    
    return computeROCAUC(merged_prediction, ground_truth)

# make the sum 1.0 and return the message
def modifySumTo1(w):
    sumWeights = sum(w)

    if sumWeights != 1.0:
        for i in range(len(w)):
            w[i] /= sumWeights

        return 'modify weight to make the sum 1.0\nmodified weight: ' + str(w) + '\n'
    else:
        return ''

# get final prediction with weights (w)
def getFinalPrediction(test_predict, w):
    
    for i in range(len(w)):
        if i == 0:
            final_prediction = w[0] * test_predict[0]
        else:
            final_prediction += w[i] * test_predict[i]

    return final_prediction

def predict(valid_output, valid_predictions, test_predictions):

    # configuration
    wcr = 0.005
    rounds = 200

    # initialize weight array
    numModel = len(valid_predictions)
    w = []
    
    for i in range(numModel):
        w.append(1 / numModel)

    # read files
    valid_ground_truth = np.array(pd.read_csv(valid_output, index_col=0))
    valid_predict = []
    test_predict = []
    
    for i in range(numModel):
        valid_predict.append(np.array(pd.read_csv(valid_predictions[i], index_col=0)))
        test_predict.append(np.array(pd.read_csv(test_predictions[i], index_col=0)))

    # log
    log = ''

    for i in range(rounds):
        print(i)
        
        score = computeMetric(valid_predict, w, valid_ground_truth)
        log += ('\n[ round ' + str(i) + ' ]\nweights=' + str(np.round_(w, 6)) + ' score=' + str(round(score, 8)) + '\n')

        # explore neighboring cases
        w_neighbors = []
        score_neighbors = []
        
        for j in range(numModel):
            w_neighbor = []

            for k in range(numModel):
                if j == k:
                    w_neighbor.append(min(w[k] + (numModel - 1)*wcr, 1.0))
                else:
                    w_neighbor.append(max(w[k] - wcr, 0.0))
            
            w_neighbors.append(w_neighbor)
            score_neighbors.append(computeMetric(valid_predict, w_neighbor, valid_ground_truth))

        # save log for each round
        f = open('log.txt', 'w')
        f.write(log)
        f.close()

        # modify weights using computed scores
        for j in range(numModel):
            log += ('neighbor' + str(j) + '=' + str(np.round_(w_neighbors[j], 6)) + ' score=' + str(round(score_neighbors[j], 8)) + '\n')
        
        # move to the neighbor with maximum score
        # stop if no neighbor with score greater than 'score'
        scores = np.array(score_neighbors)
        moveTo = scores.argmax()

        if scores[moveTo] > score:
            for i in range(len(w)):
                if i == moveTo:
                    w[i] = min(w[i] + (numModel - 1)*wcr, 1.0)
                else:
                    w[i] = max(w[i] - wcr, 0.0)
                    
            log += 'move to neighbor' + str(moveTo) + '\n'

        else:
            break

        # modify weight to make the sum 1.0
        sumWeights = sum(w)
        log += modifySumTo1(w)
        
    # save log for each round at finish
    f = open('log.txt', 'w')
    f.write(log)
    f.close()

    # make final prediction
    final_prediction = getFinalPrediction(test_predict, w)
    final_prediction = pd.DataFrame(final_prediction)
    final_prediction.to_csv('finalPrediction.csv')

if __name__ == '__main__':

    valid_output = 'bert_train_output.csv'
    valid_predictions = ['bert_valid_prediction_model_0.csv',
                         'bert_valid_prediction_model_1.csv',
                         'bert_valid_prediction_model_2.csv',
                         'bert_valid_prediction_model_3.csv',
                         'bert_valid_prediction_model_4.csv',
                         'bert_valid_prediction_model_5.csv',
                         'merged_predictions_model_0.csv',
                         'merged_predictions_model_1.csv',
                         'merged_predictions_model_2.csv',
                         'merged_predictions_model_3.csv']
    test_predictions = ['bert_test_prediction_model_0.csv',
                        'bert_test_prediction_model_1.csv',
                        'bert_test_prediction_model_2.csv',
                        'bert_test_prediction_model_3.csv',
                        'bert_test_prediction_model_4.csv',
                        'bert_test_prediction_model_5.csv',
                        'fin0_model_0.csv',
                        'fin0_model_1.csv',
                        'fin0_model_2.csv',
                        'fin0_model_3.csv']
    
    predict(valid_output, valid_predictions, test_predictions)
