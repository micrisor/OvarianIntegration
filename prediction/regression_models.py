# BASIC STUFF
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

# SKLEARN
from sklearn.linear_model import ElasticNet, Lasso, LassoCV,  BayesianRidge, LassoLarsIC
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve, auc

# SCIPY
from scipy import interp
from scipy.stats import spearmanr

def defineLabels():
    labels = {}
    labels['relative_change'] = 'Volume change'
    return labels

def optimise_rf_featsel(X, y, cut, cv=5, auc_cut=None, label='Response', prefix='someresponse'):
    # Pipeline components
    scaler = RobustScaler()
    kbest = SelectAtMostKBest(score_func=f_regression)
    dropcoll = DropCollinear(cut)
    rf = RandomForestRegressor(random_state=1)
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('rf', rf)])
    # Parameter ranges
    param_grid = { 'kbest__k': range(1,X.shape[1]),
                    "rf__max_depth": [3, None],
                    "rf__n_estimators": [5, 10, 25, 50, 100],
                    "rf__max_features": [0.05, 0.1, 0.2, 0.5, 0.7],
                    "rf__min_samples_split": [2, 3, 6, 10, 12, 15]
                    }
    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, iid=False, cv=cv, scoring='neg_mean_squared_error',return_train_score=True, n_jobs=-1, verbose=0,n_iter=3000, random_state=1)
    search.fit(X,y)
    # Best result
    best_iter = np.argmax(search.cv_results_['mean_test_score'])
    return search


def optimise_elnet_featsel(X, y, cut, cv=5, auc_cut=None, label='Response', prefix='someresponse'):
    # Pipeline components
    scaler = RobustScaler()
    kbest = SelectAtMostKBest(score_func=f_regression)
    dropcoll = DropCollinear(cut)
    elnet = ElasticNet(random_state=1, max_iter=1000000)
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('elnet', elnet)])
    # Parameter ranges
    param_grid = { 'kbest__k': range(1,X.shape[1]),
                    'elnet__alpha': np.logspace(-8, 1, 200),
                    'elnet__l1_ratio': np.arange(0.1,1.1,0.1)
                    }
    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, iid=False, cv=cv, scoring='neg_mean_squared_error', n_iter=3000,return_train_score=True, n_jobs=-1, random_state=1)
    search.fit(X,y)
    # Best result
    best_iter = np.argmax(search.cv_results_['mean_test_score'])

    return search

def optimise_svr_featsel(X, y, cut, cv=5, auc_cut=None, label='Response', prefix='someresponse'):
    # Pipeline components
    scaler = RobustScaler()
    kbest = SelectAtMostKBest(score_func=f_regression)
    dropcoll = DropCollinear(cut)
    svr = SVR(kernel='rbf', max_iter=1000000)
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('svr', svr)])
    # Parameter ranges
    param_grid = { 'kbest__k': range(1,X.shape[1]),
                    'svr__gamma': np.logspace(-9,3,100),
                    'svr__C': np.logspace(-3,3,100)
                    }
    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, iid=False, cv=cv, scoring='neg_mean_squared_error', n_iter=3000,return_train_score=True, n_jobs=-1, random_state=1)
    search.fit(X,y)
    # Best result
    best_iter = np.argmax(search.cv_results_['mean_test_score'])
    return search


def plot_and_refit(X, y, model, cv, auc_cut, label='Response', prefix='someresponse',feats='features'):
    mses = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 10)

    ypreds = []
    yreals = []
    ypreds_cv = []
    yreals_cv = []
    cv_models = []

    for i,(tr,ts) in enumerate(cv):
        model.fit(X.loc[tr,:], y.loc[tr])
        cv_models.append(deepcopy(model))
        y_pred = model.predict(X.loc[ts,:])
        y_binary = y.loc[ts]>auc_cut

        # MSE
        mses.append( mean_squared_error(y.loc[ts], y_pred) )
        ypreds.extend(y_pred)
        yreals.extend(y_binary)
        ypreds_cv.append(y_pred)
        yreals_cv.append(y_binary)

        # AUC
        fpr, tpr, thresholds = roc_curve(y_binary, y_pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    # Mean AUC
    mean_auc = np.mean(aucs)
    median_auc = np.median(aucs)
    std_auc = np.std(aucs)

    # Mean MSE
    mean_mse = np.mean(mses)
    std_mse = np.std(mses)

    f = open('/data/'+'output_'+feats+'.txt', 'a')
    f.write('{},cv,{},{},{},{},{},{},{}\n'.format(feats,prefix,label,mean_auc,median_auc,std_auc,mean_mse,std_mse))
    f.close()

    ### Refit
    model.fit(X,y)
    return [model, cv_models]


def fit_stack(X, y, rcut, kf, auc_cut=None, label='Response', prefix='someresponse'):
    n_folds = 5
    print('Elastic Net')
    elnet_res = optimise_elnet_featsel(X,y,rcut,cv=kf,auc_cut=auc_cut,label=label, prefix=prefix)
    print('SVR')
    svr_res = optimise_svr_featsel(X,y,rcut,cv=kf,auc_cut=auc_cut,label=label, prefix=prefix)
    print('RF')
    rf_res = optimise_rf_featsel(X,y,rcut,cv=kf,auc_cut=auc_cut,label=label, prefix=prefix)
    print('Averages')
    averaged_models = AveragingModels(models = (elnet_res,svr_res,rf_res))

    models = {'elnet':elnet_res, 'svr':svr_res, 'rf':rf_res, 'avg':averaged_models}

    return models


def refit_all_models(X,y,results,splits,auc_cut,whichFeats,criterion):
    refit = {}
    for model in results.keys():
        try:
            refit[model] = plot_and_refit(X,y,results[model].best_estimator_,splits,auc_cut,label=model,prefix=criterion,feats=whichFeats)
        except:
            refit[model] = plot_and_refit(X,y,results[model],splits,auc_cut,label=model,prefix=criterion,feats=whichFeats)
    return refit


###################################
######### TEST FUNCTIONS ##########
###################################

def test_all_models(X,y,results,auc_cut=None, label='Response',prefix='someresponse'):
    test_result = {}
    labels = defineLabels()
    for model in results.keys():
        test_result[model] = final_test(X,y,results[model],auc_cut=auc_cut,label=label,prefix=model+'_'+prefix)
    return test_result

def final_test(X, y, model, auc_cut=None, label='Response', prefix='someresponse'):
    y_pred = model[0].predict(X)
    return [y, y_pred]


###################################
######### AUX  FUNCTIONS ##########
###################################

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x.best_estimator_) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


### Custom class inspired by:
### https://stackoverflow.com/questions/25250654/how-can-i-use-a-custom-feature-selection-function-in-scikit-learns-pipeline
### https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/
class DropCollinear(BaseEstimator, TransformerMixin):
    def __init__(self, thresh):
        self.uncorr_columns = None
        self.thresh = thresh

    def fit(self, X, y):
        cols_to_drop = []

        # Find variables to remove
        X_corr = X.corr()
        large_corrs = X_corr>self.thresh
        indices = np.argwhere(large_corrs.values)
        indices_nodiag = np.array([[m,n] for [m,n] in indices if m!=n])

        if indices_nodiag.size>0:
            indices_nodiag_lowfirst = np.sort(indices_nodiag, axis=1)
            correlated_pairs = np.unique(indices_nodiag_lowfirst, axis=0)
            resp_corrs = np.array([[np.abs(spearmanr(X.iloc[:,m], y).correlation), np.abs(spearmanr(X.iloc[:,n], y).correlation)] for [m,n] in correlated_pairs])
            element_to_drop = np.argmin(resp_corrs, axis=1)
            list_to_drop = np.unique(correlated_pairs[range(element_to_drop.shape[0]),element_to_drop])
            cols_to_drop = X.columns.values[list_to_drop]

        cols_to_keep = [c for c in X.columns.values if c not in cols_to_drop]
        self.uncorr_columns = cols_to_keep

        return self

    def transform(self, X):
        return X[self.uncorr_columns]

    def get_params(self, deep=False):
        return {'thresh': self.thresh}

### Inspired by: https://stackoverflow.com/questions/29412348/selectkbest-based-on-estimated-amount-of-features/29412485
class SelectAtMostKBest(SelectKBest):
    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            # set k to "all" (skip feature selection), if less than k features are available
            self.k = "all"
