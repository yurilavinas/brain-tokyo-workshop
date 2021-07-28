import numpy as np
import math
import copy
import json

from domain import *  # Task environments
from utils import *
from .nsga_sort import nsga_sort
from .neat import Neat
from scipy import special


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



    for i in range(np.shape(reward)[0]):
      

      if p['alg_selection'] == "count":
        self.pop[i].count = reward[i,len(reward[i,:])-1]
        my_data = np.delete(reward[i,:], len(reward[i,:])-1) 
      else:
        my_data = reward[i,:]


      self.pop[i].fitness = np.mean(my_data)
      self.pop[i].fitMax  = np.max(my_data)
      self.pop[i].nConn   = self.pop[i].nConn
      mean_vec = np.full(shape=p['alg_nVals'],fill_value = self.pop[i].fitness)
      self.pop[i].kl_stat = np.sum(special.kl_div(np.asarray(my_data), mean_vec))
      # self.pop[i].kl_stat = np.sum(special.kl_div(np.asarray(reward[i,:]), unif_sampled))
      # self.pop[i].stat = stats.kstest(reward[i,:], stats.randint.cdf, args=(0,np.max(reward)))[1]
      self.pop[i].var = np.var(np.clip(reward[i,:], 0, max(my_data)))
      self.pop[i].rewards   = my_data
      
    # novelty = np.zeros(len(self.pop))
    # for i in range(len(self.pop)):
    #   self.pop[i].novelty = sparseness(self.archive, self.pop, self.pop[i].conn)
    #   novelty[i] = self.pop[i].novelty
    # if len(self.archive) > 0:
    #   archive_novelty = [ind.novelty for ind in self.archive]
    #   if self.pop[np.argmax(novelty)].novelty > self.archive[np.argmax(archive_novelty)].novelty:
    #     self.archive.append(copy.deepcopy(self.pop[np.argmax(novelty)]))
    #   if len(self.archive) > len(self.pop):
    #     del self.archive[0]
    # else:
    #   self.archive.append(copy.deepcopy(self.pop[np.argmax(novelty)]))
      
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

