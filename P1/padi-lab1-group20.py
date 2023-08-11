# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd

def load_chain(fName):
  X = ('1', '2', '3', '4', '5', '6', '7',
       '8', '9', '10', '11', '12', '13', '14',
       '15', '16', '17', '18', '19', '20', '21',
       '22', '23', '24', '25', '26', '27', '28',
       '29', '30', '31', '32', '33', '34', '35')
  
  P = np.load(fName)
  
  markovChain = (X, P)

  return markovChain


def prob_trajectory(markovChain, trajectory):
    P = markovChain[1]
    trajectProb = 1
    
    for i in range(len(trajectory) - 1):
      currentCell = int(trajectory[i]) - 1
      nextCell = int(trajectory[i+1]) - 1
      
      trajectProb = trajectProb * P[currentCell][nextCell]
    
    return trajectProb


def stationary_dist(markovChain):
    transposedP = markovChain[1].T
    
    eigenValues, eigenVectors = np.linalg.eig(transposedP)
    
    absoluteValues = np.abs(eigenValues - 1)
    lineIndex = np.argmin(absoluteValues)
    
    transposedEigenVectors = eigenVectors.T
    stationaryDist = np.real(transposedEigenVectors[lineIndex])
    
    stationaryDistTotalSum = np.sum(stationaryDist)
    normalizedDist = stationaryDist/stationaryDistTotalSum
    
    return normalizedDist


def compute_dist(markovChain, initialDistribution, N):
    P = np.linalg.matrix_power(markovChain[1], N)

    finalDistribution = initialDistribution.dot(P)
    
    return finalDistribution

"""
For the Markov Chain to be ergodic it needs to be irreducible and aperiodic. 
The given Markov Chain in this exercise is irreducible since it possible for any state to be reached from any other state. 
The given chain is also aperiodic since the period of all given states is 1, we can conclude this because the chain is irreducible 
and if one state has period 1, all the states from the chain have period 1 as well. 
Besides that, the chain eventually reaches the stationary distribution, we can observe this because at around 100 steps the distribution 
did not reach the stationary distribution yet, but at around 2000 steps it did.
"""


def simulate(markovChain, initialDistribution, N):
    expectedTrajectory = ()
    
    stepDistribution = initialDistribution[0]
    X = markovChain[0]
    P = markovChain[1]
    for i in range(N):
        nextCellList = np.random.choice(X, 1, p = stepDistribution)
        nextCell = nextCellList[0]
        
        expectedTrajectory = expectedTrajectory + (nextCell,)
        
        stepDistribution = P[X.index(nextCell)]
        
    return expectedTrajectory



markovChain = load_chain('pacman-big.npy')
X, P = markovChain
nS = len(X)

rnd.seed(42)

u = rnd.random((1, nS))
u = u / np.sum(u)

nSteps = 20000

states = simulate(markovChain, u, nSteps)
x = [X.index(state) for state in states]
   
n, bins, patches = plt.hist(x, rwidth = 0.5 ,bins = np.arange(nS + 1) - 0.5, 
                            density = 1, color ='green', alpha = 0.7)
   
stationaryDist = stationary_dist(markovChain)

plt.plot(X, stationaryDist, 'ro', label="Stationary Distribution")
  
plt.xlabel('State')
plt.ylabel('Frequency per State')
  
plt.title('Stationary Distribution', fontweight ="bold")
  
plt.show()