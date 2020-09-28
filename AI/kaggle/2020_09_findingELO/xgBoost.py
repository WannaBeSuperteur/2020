# df_pca_train : training data (dataFrame) [pca0 pca1 ... pcaN target]
# df_pca_test  : test data (dataFrame) [pca0 pca1 ... pcaN]
# name         : name of this validation/test
# rounding     : round to integer (True or False)
# validation   : just validating xgBoost, not for obtaining result (True or False)
# xgBoostLevel : 0 then just use xgBoost, 1 then improve accuracy (=performance)
# info         : [epochs, boostRound, earlyStoppingRound, foldCount, rateOf1s]

# ref0: https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
# ref1: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# ref2: https://www.kaggle.com/jeremy123w/xgboost-with-roc-curve
# ref3: https://swlock.blogspot.com/2019/02/xgboost-stratifiedkfold-kfold.html
def usingXgBoost(df_pca_train, df_pca_test, targetColName, name, rounding, validation, xgBoostLevel, info):

    [epochs, boostRound, earlyStoppingRounds, foldCount, rateOf1s] = info

    # initialize ROC as None
    ROC = None

    # data/target for train and test
    trainData = df_pca_train.drop([targetColName], axis=1)
    trainTarget = df_pca_train[targetColName]
    testData = df_pca_test

    # change into np.array form
    if xgBoostLevel == 0:
        trainData = np.array(trainData)
        trainTarget = np.array(trainTarget)
        testData = np.array(testData)

    # split data into train and test dataset
    # when validation is True, length of finalCvPred (k-folded training data) is 3232
    # because test_size is 0.2 -> 4040 * (1 - 0.2) = 3232
    if validation == True: # validation (using some training data as test data)
        rd = random.randint(0, 9999)
        test_size = 0.2
        xTrain, xTest, yTrain, yTest = train_test_split(trainData, trainTarget, test_size=test_size, random_state=rd)
    else:
        xTrain = trainData
        yTrain = trainTarget
        xTest = testData

    # if xgBoostLevel is 0, then fit model to train dataset
    if xgBoostLevel == 0:
        model = XGBClassifier()
        model.fit(xTrain, yTrain)

        # make prediction
        testOutput = model.predict(xTest)
        
    # if xgBoostLevel is 1, then improve performance
    # ref: https://swlock.blogspot.com/2019/02/xgboost-stratifiedkfold-kfold.html
    elif xgBoostLevel == 1:

        # print shape info
        print('\n<<< [19] shape of xTrain, xTest, yTrain and yTest ] >>>')
        if validation == True:
            print('xTrain : ' + str(len(xTrain)) + ',' + str(len(xTrain.columns)))
            print('xTest  : ' + str(len(xTest)) + ',' + str(len(xTest.columns)))
            print('yTrain : ' + str(len(yTrain)))
            print('yTest  : ' + str(len(yTest)))
        else:
            print('xTrain : ' + str(len(xTrain)) + ',' + str(len(xTrain.columns)))
            print('xTest  : ' + str(len(xTest)) + ',' + str(len(xTest.columns)))
            print('yTrain : ' + str(len(yTrain)))

        # define param first
        param = {'objective':'binary:logistic', 'random_seed':0, 'eta':0.5}

        # final result
        finalCvPred = np.zeros(len(xTrain))
        finalCvROC = 0

        # transform into matrices
        xTrainMatrix = xTrain.values
        yTrainMatrix = yTrain.values

        for i in range(epochs): # for each epoch
            rd = random.randint(0, 9999)
            stratifiedkfold = StratifiedKFold(n_splits=foldCount, random_state=rd, shuffle=True)

            # result of this time (epoch i)
            cvPred = np.zeros(len(xTrain))
            cvROC = 0

            count = 0 # for 'for' loop below

            # use k-fold method
            for trainIndex, testIndex in stratifiedkfold.split(xTrainMatrix, yTrainMatrix):
                count += 1

                # check trainIndex and testIndex
                # print('\n\n\n\n\n\ntestIndex: ' + str(testIndex[:5])) # temp

                x_Train, x_Test = xTrainMatrix[trainIndex], xTrainMatrix[testIndex]
                y_Train, y_Test = yTrainMatrix[trainIndex], yTrainMatrix[testIndex]

                # using xgb.DMatrix
                dTrain = xgb.DMatrix(x_Train, label=y_Train)
                dTest = xgb.DMatrix(x_Test, label=y_Test)

                watchList = [(dTest, 'eval', dTrain, 'train')]
                model = xgb.train(param, dTrain, num_boost_round=boostRound, evals=watchList,
                                  early_stopping_rounds=earlyStoppingRounds, verbose_eval=False)

                # make prediction
                y_Predict = model.predict(dTest, ntree_limit=model.best_ntree_limit)
                
                # print evaluation result for this epoch and count, when VALIDATION
                # when VALIDATION, round y values to show the result of validation
                if validation == True:

                    # only rateOf1s values become 1
                    #print(y_Predict[:30])
                    ySorted = sorted(y_Predict, reverse=True)
                    cutline = ySorted[int(0.5 * len(ySorted))-1] # top 50% largest value
                    
                    for j in range(len(y_Predict)):
                        if y_Predict[j] >= cutline: y_Predict[j] = 1
                        else: y_Predict[j] = 0
                    #print(sum(y_Predict))

                # evaluate ROC
                fpr, tpr, _ = roc_curve(y_Test, y_Predict)
                thisROC = auc(fpr, tpr)
                    
                print('\n<<< [20-0] xgBoost ROC result [ epoch=' + str(i) + ',count=' + str(count) + ' ] >>>')
                print('validation           : ' + str(validation))
                print('y_Predict (first 10) : ' + str(y_Predict[:10]) + ' / len=' + str(len(y_Predict)))
                print('thisROC              : ' + str(thisROC))

                # add to cvPred and cvROC
                for j in range(len(y_Predict)): cvPred[testIndex[j]] += y_Predict[j]
                cvROC += thisROC
                # print('****' + str(cvPred[:20])) # temp

            # add to final result of cvPred and cvROC
            # cvPred /= foldCount
            cvROC /= foldCount
            finalCvPred += cvPred
            finalCvROC += cvROC

            # evaluate ROC (using cvPred instead of y_Predict)
            fpr, tpr, _ = roc_curve(yTrainMatrix[:len(y_Test)], cvPred[:len(y_Test)])
            cvROC_cvPred = auc(fpr, tpr)

            # print evaluation result for this epoch
            print('\n<<< [20-1] xgBoost ROC result [ epoch=' + str(i) + ' ] >>>')
            print('cvPred (first 10) : ' + str(cvPred[:10]) + ' / len=' + str(len(cvPred)))
            print('cvROC             : ' + str(cvROC))
            print('cvROC_cvPred      : ' + str(cvROC_cvPred))

        # print evaluation result
        finalCvPred /= epochs
        finalCvROC /= epochs

        # evaluate ROC (using finalCvPred instead of y_Predict)
        fpr, tpr, _ = roc_curve(yTrainMatrix[:len(y_Test)], finalCvPred[:len(y_Test)])
        cvROC_finalCvPred = auc(fpr, tpr)
        
        print('\n<<< [20-2] xgBoost ROC result >>>')
        print('xTrain index (first 50) :\n' + str(np.array(xTrain.index)[:50]))
        print('validation              : ' + str(validation))
        print('finalCvPred (first 50)  :\n' + str(finalCvPred[:50]) + ' / len=' + str(len(finalCvPred)))
        print('finalCvROC              : ' + str(finalCvROC))
        print('cvROC_finalCvPred       : ' + str(cvROC_finalCvPred))

        # only rateOf1's values become 1 for final result
        # round y values to get the final result
        if rounding == True:
            finalSorted = sorted(finalCvPred, reverse=True)
            cutline = finalSorted[int(rateOf1s * len(finalSorted))-1] # top 'rateOf1s' largest value
                
            for i in range(len(finalCvPred)):
                if finalCvPred[i] >= cutline: finalCvPred[i] = 1
                else: finalCvPred[i] = 0

            # compute cvROC_finalCVPred (using finalCvPred instead of y_Predict)
            fpr, tpr, _ = roc_curve(yTrainMatrix[:len(y_Test)], finalCvPred[:len(y_Test)])
            cvROC_finalCvPred = auc(fpr, tpr)

            # print evaluation result
            print('\n<<< [20-3] xgBoost ROC result (after rounding) >>>')
            print('xTrain index (first 50) :\n' + str(np.array(xTrain.index)[:50]))
            print('validation              : ' + str(validation))
            print('finalCvPred (first 50)  :\n' + str(finalCvPred[:50]) + ' / len=' + str(len(finalCvPred)))
            print('cvROC_finalCvPred       : ' + str(cvROC_finalCvPred))

        # return final result (testOutput and ROC)
        testOutput = finalCvPred
        ROC = finalCvROC

        # finish when VALIDATION
        if validation == True: return

        ##### EXECUTED FOR TESTING ONLY #####
        # proceed when TESTING

        # use kNN algorithm

        # return test output
        return testOutput

    # for XG_BOOST_LEVEL 0
    # save predictions into the array
    # 'predictions' is corresponding to 'testOutput'
    predictions = []
    if rounding == True:
        for i in range(len(testOutput)): predictions.append(round(testOutput[i], 0))
    else: predictions = testOutput

    # evaluate and return predictions
    print('\n<<< [21] xgBoost test result [ ' + name + ' ] >>>')
    if validation == True:
        
        # evaluate accuracy
        accuracy = accuracy_score(yTest, predictions)

        # evaluate ROC
        fpr, tpr, _ = roc_curve(yTest, testOutput)
        if ROC == None: ROC = auc(fpr, tpr)
        
        # print evaluation result
        print('prediction (first 10) : ' + str(predictions[:10]))
        print('accuracy              : ' + str(accuracy))
        print('ROC                   : ' + str(ROC))
        
        return (accuracy, ROC, predictions)

    else:
        print('prediction (first 20) : ' + str(predictions[:20]))
        return predictions
