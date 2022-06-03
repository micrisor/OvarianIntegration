####
#### Train predictive model with a certain feature set and feature selection criteria
####

def defineSplits(X,y,n_folds,random_state):
    import numpy as np
    from sklearn.model_selection import KFold, StratifiedKFold
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=int(random_state))
    response_bins = np.percentile(y, [0,20,40,60,80,100])
    binned_response = np.digitize(y, response_bins)
    cv_indices=list(cv.split(X,binned_response))
    return cv_indices

def defineSet(df, featClass, response):
    import pickle
    import numpy as np
    from combinations import feature_combinations

    feat_classes = pickle.load(open('inputs/feature_classes.p','rb'))
    combiclass = feature_combinations(feat_classes)
    feats = combiclass.getFeatures(featClass)

    logterm = getLogSum()

    X = df[feats].copy()
    y = np.log(df[response]+logterm[response])
    nans = y.isna().values
    X = X[~nans].copy()
    y = y[~nans].copy()
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    return X, y

def getLogSum():
    logsum = {}
    logsum['relative_change'] = 0
    return logsum

def defineLabels():
    labels = {}
    labels['relative_change'] = 'Volume change'
    return labels

def plotStyle():
    from matplotlib import rcParams
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['.Helvetica Neue DeskInterface']
    rcParams['font.size'] = 18
    rcParams['axes.linewidth'] = 2
    rcParams['grid.linewidth'] = 2
    rcParams['grid.color'] = 'gainsboro'
    rcParams['font.weight'] = 'normal'
    rcParams['axes.labelweight'] = 'bold'
    rcParams['axes.labelsize'] = 21
    rcParams['legend.edgecolor'] = 'none'
    rcParams["axes.spines.right"] = False
    rcParams["axes.spines.top"] = False

def main(feats, response, rcut, prior, random_state):
## SETUP
    from regression_models import fit_stack, refit_all_models, test_all_models
    import numpy as np
    import pandas as pd
    import pickle
    plotStyle()

## PARAMETERS
    n_folds = 5
    rcut = float(rcut)

## INPUT
    df_train = pd.read_csv('inputs/training_df.csv')
    df_test = pd.read_csv('inputs/testing_df.csv')

## DATASET
    Xtrain, ytrain = defineSet(df_train, feats, response)
    splits = defineSplits(Xtrain, ytrain, n_folds=5, random_state=random_state)

## MODELS
    auc_cut = round(np.mean(ytrain),2)
    labels = defineLabels()

    # Train
    response_models = fit_stack(Xtrain, ytrain, rcut, splits, auc_cut=auc_cut, label=labels[response], prefix='{}_{}'.format(feats,response))
    with open('/data/{}_{}_{}_training.pkl'.format(feats, response, prior), 'wb') as f:
        pickle.dump(response_models, f)

    # Refit
    response_refits = refit_all_models(Xtrain, ytrain, response_models, splits, auc_cut, feats, response)
    with open('/data/{}_{}_{}_refits.pkl'.format(feats, response, prior), 'wb') as f:
        pickle.dump(response_refits, f)

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
