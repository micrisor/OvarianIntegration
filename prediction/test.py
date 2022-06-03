####
#### Test models with a certain feature set and feature selection criteria
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

def main(feats, response, rcut, rseed):
## SETUP
    import regression_models
    import paper_survival_RF
    import numpy as np
    import pandas as pd
    import pickle
    plotStyle()

## PARAMETERS
    n_folds = 5
    rcut = float(rcut)
    prior = 'basic'

## INPUT
    #df_test = pd.read_csv('inputs/testing_df.csv')
    df_test = pd.read_csv('inputs/barts_df.csv')

## DATASET
    Xtest, ytest = defineSet(df_test, feats, response)


## MODELS
    auc_cut = None
    labels = defineLabels()
    parent_folder = 'cluster_results/results_rs{}_submission_20200809_155423'.format(rseed)

    with open('{parent}_{feats}_{response}_{rcut}_{prior}/{feats}_{response}_{prior}_training.pkl'.format(parent=parent_folder, feats=feats, response=response, rcut=rcut, prior=prior), 'rb') as f:
        response_models = pickle.load(f)
    with open('{parent}_{feats}_{response}_{rcut}_{prior}/{feats}_{response}_{prior}_refits.pkl'.format(parent=parent_folder, feats=feats, response=response, rcut=rcut, prior=prior), 'rb') as f:
        response_refits = pickle.load(f)

    # Test
    test_results = regression_models.test_all_models(Xtest, ytest, response_refits, auc_cut=auc_cut, label=labels[response], prefix='{}_{}_{}'.format(feats,response,prior))
    with open('{}_{}_{}_rs{}_testresults.pkl'.format(feats, response, prior, rseed), 'wb') as f:
        pickle.dump(test_results, f)

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
