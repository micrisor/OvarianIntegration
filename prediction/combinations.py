class feature_combinations:
    def __init__(self, feature_classes):
        # cumulative combinations
        self.feats = {}
        self.feats['cum_age_stage'] = feature_classes['age_stage']
        self.feats['cum_treatment'] = self.feats['cum_age_stage'] + feature_classes['treatment']
        self.feats['cum_ca125'] = self.feats['cum_treatment'] + feature_classes['ca125']
        self.feats['cum_semantic'] = self.feats['cum_ca125'] + feature_classes['semantic']
        self.feats['cum_recist'] = self.feats['cum_semantic'] + feature_classes['recist']
        self.feats['cum_volume'] = self.feats['cum_recist'] + feature_classes['volume']
        self.feats['cum_global_radiomics'] = self.feats['cum_volume'] + feature_classes['global_radiomics']
        self.feats['cum_ratio_radiomics'] = self.feats['cum_global_radiomics'] + feature_classes['ratio_radiomics']
        self.feats['cum_ctdna'] = self.feats['cum_ratio_radiomics'] + feature_classes['ctdna']

    def getFeatures(self, combi):
        return self.feats[combi]
