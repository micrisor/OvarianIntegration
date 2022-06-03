import pickle
import numpy as np

def featurePriority(response, columns, priority_selection):
    feat_classes = pickle.load(open('inputs/feature_classes.p','rb'))

    all_class_names = ['age_stage', 'treatment', 'ca125', 'semantic', 'recist', 'volume', 'global_radiomics', 'ratio_radiomics', 'ctdna']
    class_priorities = {}
    class_priorities['basic'] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    class_priorities['replace_treatment'] = [0, 8, 1, 2, 3, 4, 5, 6, 7]
    class_priorities['replace_ca125'] = [0, 1, 8, 2, 3, 4, 5, 6, 7]
    class_priorities['replace_clinical'] = [7, 8, 5, 2, 1, 0, 3, 4, 6]

    dic_priorities = {}
    for prt,cn in zip(class_priorities[priority_selection], all_class_names):
        for fn in feat_classes[cn]:
            dic_priorities[fn] = prt

    col_priorities =  []
    for col in columns:
        col_priorities.append( dic_priorities[col] )

    return np.array(col_priorities)
