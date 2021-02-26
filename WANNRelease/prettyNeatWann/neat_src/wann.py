import numpy as np
import math
import copy
import json

from domain import *  # Task environments
from utils import *
from .nsga_sort import nsga_sort
from .neat import Neat


class Wann(Neat):
  """NEAT main class. Evolves population given fitness values of individuals.
  """
  def __init__(self, hyp):
    """Intialize NEAT algorithm with hyperparameters
    Args:
      hyp - (dict) - algorithm hyperparameters

    Attributes:
      p       - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
      pop     - (Ind)      - Current population
      species - (Species)  - Current species   
      innov   - (np_array) - innovation record
                [5 X nUniqueGenes]
                [0,:] == Innovation Number
                [1,:] == Source
                [2,:] == Destination
                [3,:] == New Node?
                [4,:] == Generation evolved
      gen     - (int)      - Current generation
    """
    Neat.__init__(self,hyp)
    self.indType = WannInd

  def tell(self,reward):
    """Assigns fitness to current population

    Args:
      reward - (np_array) - fitness value of each individual
               [nInd X nTrails]

    """

    p = self.p

    # if p['alg_selection'] == "var":


    #   for i in range(np.shape(reward)[0]):
    #     reward[i] = np.clip(reward[i],0, max(reward[i]))
    #     reward1 = reward[i][0:8]
    #     reward2 = reward[i][8:16]
    #     var1 = np.var(reward1)
    #     var2 = np.var(reward2)        
    #     self.pop[i].fitness = np.mean(reward[i,:])
    #     self.pop[i].mean = self.pop[i].fitness
    #     self.pop[i].var = var1 + var2
    #     self.pop[i].fitMax  = np.max(reward[i])
    #     self.pop[i].nConn   = self.pop[i].nConn
    #     self.pop[i].rewards   = (reward1 + reward2)/2
    # else:
    for i in range(np.shape(reward)[0]):
      self.pop[i].fitness = np.mean(np.clip(reward[i,:], 0, max(reward[i,:])))
      self.pop[i].mean = np.mean(reward[i,:])
      self.pop[i].var = np.var(np.clip(reward[i,:], 0, max(reward[i,:])))
      # if p['alg_selection'] == "novelty":
      #   self.pop[i].novelty   = self.pop[i].novelty
      self.pop[i].fitMax  = np.max(reward[i,:])
      self.pop[i].nConn   = self.pop[i].nConn
      self.pop[i].rewards   = reward

    novelty = np.zeros(len(self.pop))
    for i in range(len(self.pop)):
      self.pop[i].novelty = sparseness(self.archive, self.pop, self.pop[i].nConn)
      novelty[i] = self.pop[i].novelty
    if len(self.archive) > 0:
      archive_novelty = [ind.novelty for ind in self.archive]
      if self.pop[np.argmax(novelty)].novelty > self.archive[np.argmax(archive_novelty)].novelty:
        self.archive.append(copy.deepcopy(self.pop[np.argmax(novelty)]))
    else:
      self.archive.append(copy.deepcopy(self.pop[np.argmax(novelty)]))
      
  def probMoo(self):
      """Rank population according to Pareto dominance.
      """
      # Compile objectives
      meanFit = np.asarray([ind.fitness for ind in self.pop])
      maxFit  = np.asarray([ind.fitMax  for ind in self.pop])
      nConns  = np.asarray([ind.nConn   for ind in self.pop])
      nConns[nConns==0] = 1 # No conns is always pareto optimal (but boring)
      objVals = np.c_[meanFit,maxFit,1/nConns] # Maximize
      # Alternate second objective
      if self.p['alg_probMoo'] < np.random.rand():
        rank = nsga_sort(objVals[:,[0,1]])
      else:
        rank = nsga_sort(objVals[:,[0,2]])
      # Assign ranks
      for i in range(len(self.pop)):
        self.pop[i].rank = rank[i]

