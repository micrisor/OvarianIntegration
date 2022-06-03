####
#### Launch the training of models with a cumulative number of feature classes
####

import os

for rseed in [1,2,3,4,5]:
    os.system( 'python test_only.py cum_treatment relative_change 0.95 {}'.format(rseed) )
    os.system( 'python test_only.py cum_ca125 relative_change 0.95 {}'.format(rseed) )
    os.system( 'python test_only.py cum_ratio_radiomics relative_change 0.95 {}'.format(rseed) )
    os.system( 'python test_only.py cum_ctdna relative_change 0.95 {}'.format(rseed) )
