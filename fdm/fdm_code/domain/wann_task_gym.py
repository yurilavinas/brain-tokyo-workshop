import random
import numpy as np
import sys
from domain.make_env import make_env
from domain.task_gym import GymTask
from neat_src import *
import math


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


  def getFitness(self, wVec, aVec, hyp, \
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
      wVals = np.array((-2,2,0.5,-1.5,-1.0,-0.5,1.0,1.5))
    else:
      wVals = np.linspace(-self.absWCap, self.absWCap ,nVals)


    # Get reward from 'reps' rollouts -- test population on same seeds
    reward = np.empty((nRep,nVals))
    cos =  [[]] * nVals
    sin =  [[]] * nVals
    pos_x =  [[]] * nVals
    count =  [[]] * nVals
    
    for iRep in range(nRep):
      for iVal in range(nVals):
        wMat = self.setWeights(wVec,wVals[iVal])
        
        if seed == -1:
          reward[iRep,iVal] = self.testInd(wMat, aVec, view=view, seed=42, returnVals=returnVals)
        else:
          reward[iRep,iVal] = self.testInd(wMat, aVec, seed=42,view=view, returnVals=returnVals)
        cos[iVal] = self.cos
        sin[iVal] = self.sin
        pos_x[iVal] = self.pos_x

    

        if(hyp['alg_selection'] == "count"):

          pos_x_ = pos_x[iVal]
          cos = cos[iVal]
          sin = sin[iVal]
          
          x = np.array([i*0.6 for i in cos]) + pos_x_
          y = np.array([i*0.6 for i in sin]) + pos_x_
          
          if  x[len(x)-1] < -2.4:
            count[iVal] = -1 # -1 - exit to the left
          elif x[len(x)-1] > 2.4: 
            count[iVal] = 1 # 1 - exit to the right
          
          else:
            tmp = y > 0
            j = 0
            count[iVal] = 1
            for i in range(len(tmp)):
              if j != i:
                if tmp[j]!= tmp[i]: 
                  j = i
                  count[iVal] += 1 # couting transitions between below and above horizontal line
            count[iVal] = count[iVal]/2 # times the pole move to an upward position

            if count[iVal] == 0.5:
              count[iVal] = 0 # 0 -  success
            else:
              left = sum(x > 0) 
              right = sum(x < 0)
              if (left != right):
                idx = np.argmax(np.asarray([left, right]))
                if (idx == 0):
                  count[iVal] = -2 # -2 - looping to the left
                else:
                  count[iVal] = 2 # 2 - looping to the right
              else:
                count[iVal] = 3 # 3 - looping both ways, equally

    if returnVals is True:
      return np.mean(reward,axis=0), np.std(reward,axis=0), wVals, cos, sin, pos_x
    elif(hyp['alg_selection'] == "count"):  
      count = len(np.unique(count))
      return np.mean(reward,axis=0), count
    else:
      return np.mean(reward,axis=0)

