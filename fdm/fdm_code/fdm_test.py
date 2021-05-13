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
  fitness, std, \
  wVals, ang_pos, pos_x = task.getFitness(wVec, aVec, hyp,
                                nVals=nMean, nRep=nRep,\
                                view=view,returnVals=True, seed=seed)      

  for i in range(nMean):

    lsave('behaviour/'+outPref+'_ang_value_'+str(i)+'.out',np.hstack(ang_pos[i]))    
    lsave('behaviour/'+outPref+'_pos_x_'+str(i)+'.out',np.hstack(pos_x[i]))
    
  lsave('test_log/'+outPref+'_std.out',std)
  lsave('test_log/'+outPref+'_wVals.out',wVals)

  
  
  # lsave('behaviour/'+outPref+'_std.out',std)
  # lsave('behaviour/'+outPref+'_vel_0.out',vel_0)
  # lsave('behaviour/'+outPref+'_vel_25.out',vel_25)
  # lsave('behaviour/'+outPref+'_vel_50.out',vel_50)
  # lsave('behaviour/'+outPref+'_vel_75.out',vel_75)
  # lsave('behaviour/'+outPref+'_vel_100.out',vel_100)
  # lsave('behaviour/'+outPref+'_pos_x_0.out',pos_x_0)
  # lsave('behaviour/'+outPref+'_pos_x_25.out',pos_x_25)
  # lsave('behaviour/'+outPref+'_pos_x_50.out',pos_x_50)
  # lsave('behaviour/'+outPref+'_pos_x_75.out',pos_x_75)
  # lsave('behaviour/'+outPref+'_pos_x_100.out',pos_x_100)
  
  # lsave('behaviour/'+outPref+'_pos_y_0.out',pos_y_0)
  # lsave('behaviour/'+outPref+'_pos_y_25.out',pos_y_25)
  # lsave('behaviour/'+outPref+'_pos_y_50.out',pos_y_50)
  # lsave('behaviour/'+outPref+'_pos_y_75.out',pos_y_75)
  # lsave('behaviour/'+outPref+'_pos_y_100.out',pos_y_100)
  # lsave('behaviour/'+outPref+'_reward.out',fitness)
  # lsave('behaviour/'+outPref+'_wVals.out',wVals)
  
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
  
