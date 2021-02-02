import random
import numpy as np
import sys
from domain.make_env import make_env
from neat_src import *
import gym


class GymTask():
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
    # Network properties
    self.nInput   = game.input_size
    self.nOutput  = game.output_size      
    self.actRange = game.h_act
    self.absWCap  = game.weightCap
    self.layers   = game.layers      
    self.activations = np.r_[np.full(1,1),game.i_act,game.o_act]
  
    # Environment
    self.nReps = nReps
    self.maxEpisodeLength = game.max_episode_length
    self.actSelect = game.actionSelect
    # if not paramOnly:
    #   self.env = make_env(game.env_name)
    
    # Special needs...
    # self.needsClosed = False# (game.env_name.startswith("CartPoleSwingUp"))    
  
  # def getFitness(self, wVec, aVec, game, view=False, nRep=False, seed=-1):
  #   """Get fitness of a single individual.
  
  #   Args:
  #     wVec    - (np_array) - weight matrix as a flattened vector
  #               [N**2 X 1]
  #     aVec    - (np_array) - activation function of each node 
  #               [N X 1]    - stored as ints (see applyAct in ann.py)
  
  #   Optional:
  #     view    - (bool)     - view trial?
  #     nReps   - (nReps)    - number of trials to get average fitness
  #     seed    - (int)      - starting random seed for trials
  
  #   Returns:
  #     fitness - (float)    - mean reward over all trials
  #   """
  #   if nRep is False:
  #     nRep = self.nReps
  #   wVec[np.isnan(wVec)] = 0
  #   reward = np.empty(nRep)

  #   # env = make_env(game.env_name)
  #   for iRep in range(nRep):
  #     # if seed > 0:
  #     #   seed = seed+iRep
      
  #     reward[iRep] = self.testInd(wVec, aVec, game, env, view=view)
    
  #   # env.close()
  #   return fitness

  def testInd(self, wVec, aVec, game, folder=None, hyp=None, view=False, seed=-1):
    """Evaluate individual on task
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
  
    Optional:
      view    - (bool)     - view trial?
      seed    - (int)      - starting random seed for trials
  
    Returns:
      fitness - (float)    - reward earned in trial
    """
    env = make_env(game.env_name, seed)

    # if seed >= 0:
    #   random.seed(4)
    #   np.random.seed(4)
      # self.env.seed(4)
    
    
    if folder != None:
      env.render()
      # env = gym.wrappers.Monitor(env, "recording_" + str(game.env_name) + "_" + folder, force=True)
    # env.close()
    state = env.reset()
    # self.env.t = 0
    annOut = act(wVec, aVec, self.nInput, self.nOutput, state)  
    action = selectAct(annOut,self.actSelect)    
    state, reward, done, info = env.step(action)
    if self.maxEpisodeLength == 0:
      if view:
        # if self.needsClosed:
        #   env.render(close=done)  
        # else:
          env.render()
      return reward
    else:
      totalReward = reward
    for tStep in range(self.maxEpisodeLength): 
      annOut = act(wVec, aVec, self.nInput, self.nOutput, state) 
      action = selectAct(annOut,self.actSelect) 
      state, reward, done, info = env.step(action)
      totalReward += reward  
      if view:
        # if self.needsClosed:
        #   env.render(close=done)  
        # else:
          env.render()
      if done:
        break

    # if folder != None:
    env.close() 
    
    return totalReward
