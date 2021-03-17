"""View and record the performance of a WANN with various weight values.

TODO: Parallelize evaluation

"""

import numpy as np
import argparse
import sys

np.set_printoptions(precision=2) 
np.set_printoptions(linewidth=160)

from neat_src import * # NEAT and WANNs
from domain import *   # Task environments

def main(argv):
  infile  = args.infile
  outPref = args.outPref
  hyp_default = args.default
  hyp_adjust  = args.hyperparam
  nMean   = args.nVals
  nRep    = args.nReps
  view    = args.view
  seed    = args.seed

  # Load task and parameters
  hyp = loadHyp(pFileName=hyp_default)
  updateHyp(hyp,hyp_adjust)
  task = WannGymTask(games[hyp['task']], nReps=hyp['alg_nReps'])

  # Bullet needs some extra help getting started
  if hyp['task'].startswith("bullet"):
    task.env.render("human")

  # Import individual for testing
  wVec, aVec, wKey = importNet(infile)

  # Show result
  fitness, std, pos_0, pos_25, pos_50, pos_75, pos_100, wVals = task.getFitness(wVec, aVec, hyp,
                                nVals=nMean, nRep=nRep,\
                                view=view,returnVals=True, seed=1)      
  print("[***]\tFitness:", fitness , '\n' + "[***]\tWeight Values:\t" , wVals) 
  print("[***]\tpos_0:", pos_0) 
  print("[***]\tpos_25:", pos_25) 
  print("[***]\tpos_50:", pos_50)
  print("[***]\tpos_75:", pos_50) 
  print("[***]\tpos_100:", pos_100) 

  lsave('test_log/'+outPref+'_std.out',std)
  lsave('test_log/'+outPref+'_pos_0.out',pos_0)
  lsave('test_log/'+outPref+'_pos_25.out',pos_25)
  lsave('test_log/'+outPref+'_pos_50.out',pos_50)
  lsave('test_log/'+outPref+'_pos_75.out',pos_75)
  lsave('test_log/'+outPref+'_pos_100.out',pos_100)
  lsave('test_log/'+outPref+'_reward.out',fitness)
  lsave('test_log/'+outPref+'_wVals.out',wVals)
  
# -- --------------------------------------------------------------------- -- #
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
  ''' Parse input and launch '''
  parser = argparse.ArgumentParser(description=('Test ANNs on Task'))
    
  parser.add_argument('-i', '--infile', type=str,\
   help='file name for genome input', default='log/test_best.out')

  parser.add_argument('-o', '--outPref', type=str,\
   help='file name prefix for result input', default='log/result_')
  
  parser.add_argument('-d', '--default', type=str,\
   help='default hyperparameter file', default='p/default_wann.json')

  parser.add_argument('-p', '--hyperparam', type=str,\
   help='hyperparameter file', default=None)

  parser.add_argument('-n', '--nVals', type=int,\
   help='Number of weight values to test', default=6)
   
  parser.add_argument('-r', '--nReps', type=int,\
   help='Number of repetitions', default=1)

  parser.add_argument('-v', '--view', type=str2bool,\
   help='Visualize trial?', default=False)

  parser.add_argument('-s', '--seed', type=int,\
   help='random seed', default=-1)

  args = parser.parse_args()
  main(args)                             
  
