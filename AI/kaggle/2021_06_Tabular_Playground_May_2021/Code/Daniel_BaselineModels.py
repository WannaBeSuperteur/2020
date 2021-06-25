import warnings

from sklearn.exceptions import ConvergenceWarning

from lightgbm.sklearn import LGBMClassifier
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

def read_data():
    train = pd.read_csv("/home/daniel/tabular-playground-series-may-2021/tabular-playground-series-may-2021/train.csv")
    test = pd.read_csv("/home/daniel/tabular-playground-series-may-2021/tabular-playground-series-may-2021/test.csv")
    sub = pd.read_csv("/home/daniel/tabular-playground-series-may-2021/tabular-playground-series-may-2021/sample_submission.csv")
    return train, test, sub


def get_base_models():
    base_models = []
    base_models.append(('LR'  , LogisticRegression()))
    base_models.append(('LDA' , LinearDiscriminantAnalysis()))
    base_models.append(('KNN' , KNeighborsClassifier()))
    base_models.append(('DTC' , DecisionTreeClassifier()))
    base_models.append(('NB'  , GaussianNB()))
    base_models.append(('SVM' , SVC(probability=True, max_iter=100)))
    base_models.append(('AB'  , AdaBoostClassifier()))
    base_models.append(('RF'  , RandomForestClassifier()))
    base_models.append(('ET'  , ExtraTreesClassifier()))
    base_models.append(('LGBM', LGBMClassifier()))
    return base_models


def get_scaled_models(scaler_name, outliers_removed=False):
    if scaler_name == 'standard':
        scaler = StandardScaler()
    elif scaler_name =='minmax':
        scaler = MinMaxScaler()

    if outliers_removed:
        scaler_name = "cleaned_" + scaler_name
    pipelines = []
    pipelines.append((scaler_name+'LR'  , Pipeline([('Scaler', scaler), ('LR'  , LogisticRegression())])))
    pipelines.append((scaler_name+'LDA' , Pipeline([('Scaler', scaler), ('LDA' , LinearDiscriminantAnalysis())])))
    pipelines.append((scaler_name+'KNN' , Pipeline([('Scaler', scaler), ('KNN' , KNeighborsClassifier())])))
    pipelines.append((scaler_name+'DTC' , Pipeline([('Scaler', scaler), ('DTC', DecisionTreeClassifier())])))
    pipelines.append((scaler_name+'NB'  , Pipeline([('Scaler', scaler), ('NB'  , GaussianNB())])))
    pipelines.append((scaler_name+'SVM' , Pipeline([('Scaler', scaler), ('SVM' , SVC(probability=True, max_iter=100))])))
    pipelines.append((scaler_name+'AB'  , Pipeline([('Scaler', scaler), ('AB'  , AdaBoostClassifier())])))
    pipelines.append((scaler_name+'RF'  , Pipeline([('Scaler', scaler), ('RF'  , RandomForestClassifier())])))
    pipelines.append((scaler_name+'ET'  , Pipeline([('Scaler', scaler), ('ET'  , ExtraTreesClassifier())])))
    pipelines.append((scaler_name+'LGBM', Pipeline([('Scaler', scaler), ('LGBM', LGBMClassifier())])))
    return pipelines


def compute_outliers(df, feature):
    #calculate upper and lower limits
    upper_limit = df[feature].mean() + 3 * df[feature].std()
    lower_limit = df[feature].mean() - 3 * df[feature].std()
    #select outliers
    outliers_df = df[~((df[feature] < upper_limit) & (df[feature] > lower_limit))]
    no_outliers_df = df[(df[feature] < upper_limit) & (df[feature] > lower_limit)]
    return outliers_df, no_outliers_df


def remove_outliers(df):
    tmp_df = df.copy()
    outliers_idx = []
    for column in tmp_df.columns[:-1]:
        outliers_df, _ = compute_outliers(tmp_df, column)
        tmp_idx = list(outliers_df.index)
        for idx in tmp_idx:
            if idx not in outliers_idx:
                outliers_idx.append(idx)
    return tmp_df.drop(outliers_idx)


def evaluate_models(X_train, y_train, models):
    # Test options and evaluation metric
    num_folds = 10
    scoring = "neg_log_loss"
    results = []
    model_names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=num_folds, random_state=42, shuffle=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            cv_results = -cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        model_names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    return model_names, results


def make_score_df(model_names, results):
    def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}"
        return float(prc.format(f_val))

    scores = []
    for r in results:
        scores.append(floatingDecimals(r.mean(), 4))
    score_df = pd.DataFrame({'Model':model_names, 'Score': scores})
    return score_df


if __name__ == "__main__":
    train, test, sub = read_data()

    X_train = train.drop("target", axis=1)
    y_train = train["target"]

    # No Scaling
    models = get_base_models()
    model_names, results = evaluate_models(X_train, y_train, models)
    baseline_score_df = make_score_df(model_names, results)
    print(baseline_score_df)

    # Standard Scaler
    models = get_scaled_models('standard')
    model_names, results = evaluate_models(X_train, y_train, models)
    standard_scaled_models_score = make_score_df(model_names, results)
    scores_df = pd.concat([baseline_score_df,
                           standard_scaled_models_score], axis=1)
    print(scores_df)

    # MixMax Scaler
    models = get_scaled_models('minmax')
    model_names, results = evaluate_models(X_train, y_train, models)
    minmax_scaled_models_scores = make_score_df(model_names, results)
    scores_df = pd.concat([baseline_score_df,
                           standard_scaled_models_score,
                           minmax_scaled_models_scores], axis=1)
    print(scores_df)

    # Removing outliers
    clean_df = remove_outliers(train)
    X_train_c = clean_df.drop("target", axis=1)
    y_train_c = clean_df["target"]

    models = get_scaled_models('minmax', outliers_removed=True)
    model_names, results = evaluate_models(X_train_c, y_train_c, models)
    minmax_scaled_models_scores_c = make_score_df(model_names,results)
    scores_df = pd.concat([baseline_score_df,
                           standard_scaled_models_score,
                           minmax_scaled_models_scores,
                           minmax_scaled_models_scores_c], axis=1)
    print(scores_df)

    scores_df.to_csv("output/baseline_model_scores_outliers_three_sigma.csv")