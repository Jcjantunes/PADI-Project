# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rand
import copy
import time
import math

def load_mdp(fileName, g):
  f = np.load(fileName)
    
  MDP = ()
  for array in f:
    if array == "c":
      MDP = MDP + (f[array],)
    else:
      MDP = MDP + (tuple(f[array]),) 
  MDP = MDP + (g,) 
    
  return MDP


def noisy_policy(MDP, a, eps):
  nX = len(MDP[0])
  nA = len(MDP[1])
  actionProb = 1 - eps
  otherActionProb = ((eps)/(nA - 1))
    
  policy = np.zeros((nX, nA))
  for state in range(nX):
    for action in range(nA):
      if action == a:
        policy[state][action] = actionProb
      else:
        policy[state][action] = otherActionProb
                
  return policy


def evaluate_pol(MDP, policy):
  nX = len(MDP[0])
  nA = len(MDP[1])
  P = MDP[2]
  c = MDP[3]
  g = MDP[4]
        
  policyTransitionProb = np.zeros((nX,nX))
    
  for action in range(nA):
    policyTransitionProb +=  policy[:, action] * P[action]
            
  policyCosts = policy * c
  policyCosts = policyCosts.sum(axis = 1)
    
  stateIdendityMatrix = np.identity(nX)
  J = np.linalg.inv(stateIdendityMatrix - (g * policyTransitionProb)).dot(policyCosts)
  J = np.reshape(J,(nX,1))
    
  return J


def value_iteration(MDP):
  nX = len(MDP[0])
  nA = len(MDP[1])
  P = MDP[2]
  c = MDP[3]
  g = MDP[4]

  J = np.zeros((nX, 1))
  T = np.zeros((nX, 1))
  
  err = math.inf
  i = 0
  startTime = time.time()

  while math.pow(10, -8) <= err:
    for action in range(nA):
      T = np.reshape(c[:, action], (nX, 1)) + g * P[action].dot(J)
      
      if action == 0:
        JMin = copy.deepcopy(T)          
      else:
        JMin = np.min((JMin, T), axis = 0)
                
    err = np.linalg.norm(JMin - J)
    J = JMin
    i = i + 1
  
  endTime = time.time()

  executionTime = round(endTime - startTime, 3)
  print("Execution time: " + str(executionTime) + " seconds")
  print("N. iterations: " + str(i))
  return J


def policy_iteration(MDP):
  nX = len(MDP[0])
  nA = len(MDP[1])
  P = MDP[2]
  c = MDP[3]
  g = MDP[4]
    
  pi = np.ones((nX, nA)) / nA
  quit = False
    
  i = 0
  startTime = time.time()
  while not quit:
    cpi = 0
    Ppi = 0
    Q = ()
  
    for action in range(nA):
      cpi = cpi + np.diag(pi[:, action]).dot(np.reshape(c[:, action],(nX, 1)))
      Ppi = Ppi + np.diag(pi[:, action]).dot(P[action])
            
    J = np.linalg.inv(np.eye(nX) - g * Ppi).dot(cpi)

    for action in range(nA):
      Q += np.reshape(c[:, action], (nX, 1)) + g * P[action].dot(J),
      if action == 0:
        JMin = copy.deepcopy(Q[action])          
      else:
        JMin = np.min((JMin, Q[action]), axis = 0)
        
    pinew = np.zeros((nX, nA))
    for a in range(nA):
      pinew[:, a, None] = np.isclose(Q[a], JMin, atol = math.pow(10, -8), rtol = math.pow(10, -8)).astype(int)
            
    pinew = pinew / np.sum(pinew, axis = 1, keepdims = True)

    quit = (pi == pinew).all()
    pi = pinew
    i = i + 1
    
  executionTime = round(time.time() - startTime, 3)
  print("Execution time: " + str(executionTime) + " seconds")
  print("N. iterations: " + str(i))
  return pi


NRUNS = 100
def simulate(MDP, policy, x0, length):
  nX = len(MDP[0])
  nA = len(MDP[1])
  P = MDP[2]
  c = MDP[3]
  g = MDP[4]
  
  simulationCost = 0
  for i in range(NRUNS):
    cSum = 0
    currentState = x0
    
    for l in range(length):
      selectedAction = rand.choice(nA, p = policy[currentState])
      cSum = cSum + np.power(g, l) * c[currentState][selectedAction]
      currentState = rand.choice(nX, p = P[selectedAction][currentState])
    
    simulationCost = simulationCost + cSum 
  
  return simulationCost/NRUNS