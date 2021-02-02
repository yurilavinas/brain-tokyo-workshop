import random
import numpy as np
import sys
from domain.make_env import make_env
from domain.task_gym import GymTask
from neat_src import *


class WannGymTask(GymTask):
  """Problem domain to be solved by neural network. Uses OpenAI Gym patterns.
  """ 
  def __init__(self, game, paramOnly=False, nReps=1): 
    """Initializes task environment
  
    Args:
      game - (string) - dict key of task to be solved (see domain/config.py)
  
    Optional:
      paramOnly - (bool)  - only load parameters instead of launching task?
      nReps     - (nReps) - number of trials to get average fitness
    """
    GymTask.__init__(self, game, paramOnly, nReps)


# -- 'Weight Agnostic Network' evaluation -------------------------------- -- #
  def setWeights(self, wVec, wVal):
    """Set single shared weight of network
  
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      wVal    - (float)    - value to assign to all weights
  
    Returns:
      wMat    - (np_array) - weight matrix with single shared weight
                [N X N]
    """
    # Create connection matrix
    wVec[np.isnan(wVec)] = 0
    dim = int(np.sqrt(np.shape(wVec)[0]))    
    cMat = np.reshape(wVec,(dim,dim))
    cMat[cMat!=0] = 1.0

    # Assign value to all weights
    wMat = np.copy(cMat) * wVal 
    return wMat


  def getFitness(self, wVec, aVec, hyp, game,\
                    nRep=False,seed=-1, nVals=8,view=False,returnVals=False):
    """Get fitness of a single individual with distribution of weights
  
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
      hyp     - (dict)     - hyperparameters
        ['alg_wDist']        - weight distribution  [standard;fixed;linspace]
        ['alg_absWCap']      - absolute value of highest weight for linspace
  
    Optional:
      seed    - (int)      - starting random seed for trials
      nReps   - (int)      - number of trials to get average fitness
      nVals   - (int)      - number of weight values to test

  
    Returns:
      fitness - (float)    - mean reward over all trials
    """
    if nRep is False:
      nRep = hyp['alg_nReps']

    # Set weight values to test WANN with
    if (hyp['alg_wDist'] == "standard") and nVals==8: # Double, constant, and half signal 
      wVals = np.array((-2,-1.5,-1.0,-0.5,0.5,1.0,1.5,2))
    else:
      wVals = np.linspace(-self.absWCap, self.absWCap ,nVals)


    # Get reward from 'reps' rollouts -- test population on same seeds
    tmp1 = np.empty((nRep,nVals))
    tmp2 = np.empty((nRep,nVals))
    reward = np.empty((nRep,nVals))

    # envs = list()
    # for i in range(0,nVals):
    #   envs.append(make_env(game.env_name))
    for iRep in range(nRep):
      for iVal in range(nVals):
        wMat = self.setWeights(wVec,wVals[iVal])
        if view == False:
            tmp1[iRep,iVal] = self.testInd(wMat, aVec, game, folder = None, view=view, seed=4)
        else:
          seed = np.random.randint(1, 1000000000)
          tmp1[iRep,iVal] = self.testInd(wMat, aVec, game, folder = str(iRep)+"_"+str(iVal), view=view, seed = seed)
        if view == False:
            tmp2[iRep,iVal] = self.testInd(wMat, aVec, game, folder = None, view=view, seed=72456)
        else:
          seed = np.random.randint(1, 1000000000)
          tmp2[iRep,iVal] = self.testInd(wMat, aVec, game, folder = str(iRep)+"_"+str(iVal), view=view, seed =seed)

        reward[iRep,iVal] = (tmp1[iRep,iVal] + tmp2[iRep,iVal])/2
        # if seed == -1:
        #   if view == False:
        #     reward[iRep,iVal] = self.testInd(wMat, aVec, game, folder = None, view=view, seed=seed)
        #   else:
        #     reward[iRep,iVal] = self.testInd(wMat, aVec, game, folder = str(iRep)+"_"+str(iVal), view=view, seed=seed)
        #     # reward[iRep,iVal] = self.testInd(wMat, aVec, game, env=envs[iVal], folder = str(iRep)+"_"+str(iVal), view=view, seed=seed)
        # else:
        #   if view == False:
        #     reward[iRep,iVal] = self.testInd(wMat, aVec, game, folder = None, view=view, seed=seed + irep)
        #   else:
        #     reward[iRep,iVal] = self.testInd(wMat, aVec, game, folder = str(iRep)+"_"+str(iVal), view=view, seed=seed + irep)
        # envs[iVal].close()

    if returnVals is True:
      return np.mean(reward,axis=0), wVals
    return np.mean(reward,axis=0)
 

