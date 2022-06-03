def submission(feats, resp, rcut, prior, stamp, rs):
    script='''#!/bin/bash
#SBATCH -J results_rs{rs}_{stamp}_{feats}_{resp}_{rcut}_{prior}
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=email@email.com
#SBATCH -p general
#SBATCH -o logs/slurm_rs{rs}_{stamp}_{feats}_{resp}_{rcut}_{prior}.out
#SBATCH -e logs/slurm_rs{rs}_{stamp}_{feats}_{resp}_{rcut}_{prior}.err

singularity exec -H /HOME-FOLDER:/home -B /RESULTS-FOLDER/results_rs{rs}_{stamp}_{feats}_{resp}_{rcut}_{prior}:/data anaconda.img python train.py {feats} {resp} {rcut} {prior} {rs}
'''.format(feats=feats, resp=resp, stamp=stamp, rcut=rcut, prior=prior, rs=rs)
    return script

def main():
    import time
    feats = ['cum_treatment', 'cum_ca125', 'cum_ratio_radiomics', 'cum_ctdna']
    response = ['relative_change']
    rcut = [0.95]
    priority = ['basic']
    parameters = [feats, response, rcut, priority]

    import os, itertools, time
    parameter_combinations = list(itertools.product(*parameters))

    import datetime
    stamp = 'submission_{date:%Y%m%d_%H%M%S}'.format( date=datetime.datetime.now() )

    flog = open('submissions/log_'+stamp+'.txt', 'w')
    explanation = input('What is this submission about? \n')
    flog.write(explanation)
    flog.close()

    random_states = [1,2,3,4,5]

    for rs in random_states:
        for i,combi in enumerate(parameter_combinations):
            feat_i, resp_i, rcut_i, prior_i = combi
            outdir = 'results_rs{}_{}_{}_{}_{}_{}'.format(rs, stamp, feat_i, resp_i, rcut_i, prior_i)
            if not os.path.exists('/RESULTS-FOLDER/{}'.format(outdir)):
                os.makedirs('/RESULTS-FOLDER/{}'.format(outdir))
                print('Making: /RESULTS-FOLDER/{}'.format(outdir))

            script = submission(feat_i, resp_i, rcut_i, prior_i, stamp, rs)

            scriptName = 'submissions/submit_rs{}_{}_{}_{}_{}_{}'.format(rs, stamp, feat_i, resp_i, rcut_i, prior_i)
            print(' Script name: '+scriptName)

            f = open(scriptName, 'w')
            f.write(script)
            f.close()

            print('  Running: sbatch '+scriptName)
            os.system('sbatch '+scriptName)
            time.sleep(0.2)

if __name__=='__main__':
    main()
