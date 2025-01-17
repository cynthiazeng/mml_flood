#split training and testing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import xgboost as xgb
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import pickle

def run_xgb(x_train, y_train, x_test):
    print('running xgb...')
    gs_metric = 'roc_auc'
    cv_folds = 3
    param_grid = {'max_depth': [4,6], #8
                  'learning_rate':[0.1],   #, 0.3],
                  'n_estimators': [100], #150
                 'scale_pos_weight': [10, 20, 65]
                # 'scale_pos_weight': [65]
                 }
    print(param_grid['scale_pos_weight'])
    est = xgb.XGBClassifier(verbosity=0)

    #grid search
    gs = GridSearchCV(estimator = est, param_grid=param_grid, scoring=gs_metric, cv= cv_folds, verbose=0)
    gs.fit(x_train, y_train)

    #training auc
    print("Train AUC: ", metrics.roc_auc_score(y_train, gs.predict_proba(x_train)[:,1]))

    print(gs.best_params_)

    y_pred_prob = gs.predict_proba(x_test)
    y_pred = gs.predict(x_test)

    return y_pred, y_pred_prob[:,1]

#try classification
def run_logreg(x_train, y_train, x_test):
    print('running log reg...')
    gs_metric = 'roc_auc'
    cv_folds = 3
    param_grid = {'C': np.arange(0.0, 1, 0.2), 'penalty': ['l2','l1']}
    est = LogisticRegression(random_state=42, class_weight='balanced')
    #grid search
    gs = GridSearchCV(estimator = est, param_grid=param_grid, scoring=gs_metric, cv= cv_folds, verbose=0)
    gs.fit(x_train, y_train)

    #training auc
    print("Train AUC: ", metrics.roc_auc_score(y_train, gs.predict_proba(x_train)[:,1]))

    y_pred_prob = gs.predict_proba(x_test)
    y_pred = gs.predict(x_test)

    return y_pred, y_pred_prob[:,1]

#do light gbm


#classification scores
def get_scores_clf(y_true, y_pred_prob):
    #get f1 based on different threshold
    f1_scores = []
    thres_list = [0.4,0.5,0.6,0.7]
    for thres in  thres_list:
        # From our classification prediction
        y_pred = (y_pred_prob >= thres).astype(int)
        f1= metrics.f1_score(y_true, y_pred, average='macro')
        f1_scores.append(f1)
    max_f1 = max(f1_scores)
    max_thres = thres_list [f1_scores.index(max_f1)]
    print('maximum f1 score, thres',max_f1 , max_thres )

    #get y_pred based on best threshold
    y_pred = (y_pred_prob >= max_thres).astype(int)
    # get rest of the scores
    accu = metrics.accuracy_score(y_true, y_pred)
    accu_bl = metrics.balanced_accuracy_score(y_true, y_pred)
    auc =  metrics.roc_auc_score(y_true, y_pred_prob, multi_class = 'ovo')
    #precision and recall scores
    precision = metrics.average_precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    print('auc, f1, accu, accu_bl, precision, recall= ', auc, f1, accu, accu_bl, precision, recall )
    print(metrics.confusion_matrix(y_true, y_pred))
    return auc, max_f1, accu, accu_bl, precision, recall
