# -*- coding: utf-8 -*-
import numpy as np
import math
import copy
import numpy.random as rand

def load_pomdp(fileName, g):
  f = np.load(fileName)
    
  POMDP = ()
  for array in f:
    if array == "c":
      POMDP = POMDP + (f[array],)
    else:
      POMDP = POMDP + (tuple(f[array]),) 
  
  POMDP = POMDP + (g,) 
    
  return POMDP


def gen_trajectory(POMDP, x0, n):
  nX = len(POMDP[0])
  nA = len(POMDP[1])
  nZ = len(POMDP[2])
  P = POMDP[3]
  O = POMDP[4]
    
  actionTrajectory = np.zeros(n).astype(int)
  stateTrajectory = np.zeros(n+1).astype(int)
  observationTrajectory = np.zeros(n).astype(int)

  for i in range (n+1):
    if i == 0:
      selectedState = x0
      stateTrajectory[i] = selectedState
    
    else:
      selectedAction = np.random.choice(nA)
      actionTrajectory[i-1] = selectedAction
      
      selectedState = np.random.choice(nX, 1, p = P[selectedAction][selectedState])[0]
      stateTrajectory[i] = selectedState
      
      selectedObservation = np.random.choice(nZ, 1, p = O[selectedAction][selectedState])[0]
      observationTrajectory[i-1] = selectedObservation
      
  t = (stateTrajectory, actionTrajectory, observationTrajectory)

  return t


def belief_update(POMDP, belief, action, observation):
  P = POMDP[3]
  O = POMDP[4]

  numerator = (belief.dot(P[action])).dot(np.diag(O[action][:, observation]))
  updatedBelief = numerator / np.sum(numerator)

  return updatedBelief


def sample_beliefs(POMDP, n):
  nX = len(POMDP[0])
  
  x0 = np.random.choice(nX)
  
  t = gen_trajectory(POMDP, x0, n)
  actionTrajectory = t[1] 
  observationTrajectory = t[2]
    
  timeStepBelief = np.ones((1, nX)) / nX 
  beliefs = (timeStepBelief,)
    
  for i in range(n):
    timeStepBelief = belief_update(POMDP, timeStepBelief, actionTrajectory[i], observationTrajectory[i])
    
    ignore = 0
    for b in beliefs:
      
      difference = np.linalg.norm(timeStepBelief - b)
      if (difference < math.pow(10, -3)): 
        ignore = 1
        break
    
    if (ignore == 0):
      beliefs = beliefs + (timeStepBelief,)
        
  return beliefs


def solve_mdp(POMDP):
  nX = len(POMDP[0])
  nA = len(POMDP[1])
  P = POMDP[3]
  c = POMDP[5]
  g = POMDP[6]
    
  pi = np.ones((nX, nA)) / nA
  quit = False

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
  
  QStar = np.transpose(Q)[0]
  
  return QStar


def get_heuristic_action(beliefState, Q, heuristic):
  if heuristic == "mls":
    state = np.argmax(beliefState[0])
    action = np.argmin(Q[state])
                
  elif heuristic == "av":
    votes = [0] * np.size(Q,1)
    for state in range(len(beliefState[0])):
      action = np.argmin(Q[state])
      votes[action] += beliefState[0][state]
        
    action = np.argmax(votes)

  elif heuristic == "q-mdp":
    resize = np.size(Q,1)
    for a in range(resize):
      
      if a == 0:
        min = beliefState[0].dot(Q[:, a])
        action = 0
      else:
        if beliefState[0].dot(Q[:, a]) < min:
          min = beliefState[0].dot(Q[:, a])
          action = a
    
  return action


def get_optimal_action(beliefState, av, ai):
  
  resize = np.size(av,1)
  for k in range(resize):   

    sum = np.sum(beliefState[0].dot(av[:, k]))    
    if k == 0:
      min = sum
      optimalActionIndex = ai[k]
    else:
      if sum < min:
        min = sum
        optimalActionIndex = ai[k]
          
  return optimalActionIndex