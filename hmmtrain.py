# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/EmilioEsposito/hmm-hidden-markov-model
All probabilities calculated use log-probabilities, EXCEPT for the inital dfs: emit, trans, prior
This avoids underflow through repeated divisions over thousands of iterations
We must use addition and subtraction of log-probabilities instead of multiplication/division, and use exp() to extract a result
logsumexp() sums an array of log-probabilities and converts to regular probability
"""

import pandas as pd
import numpy as np
from scipy.special import logsumexp
import re
import sys
import random
import os
numStates = int(sys.argv[2])
states = []
for i in range(1, numStates+1):
    states.append("s" + str(i))
#probability of starting in each state - generate randomly
priors = np.ndarray.flatten(np.random.dirichlet(np.ones(numStates), size=1))
prior = pd.DataFrame(priors, index = states, columns = ["prob"])
filename = ""
if os.name == 'nt':
    filename = (os.getcwd()+ "\\" + sys.argv[1])
else:
    filename = (sys.argv[1])
#load observations (words in corpus)
with open(filename, 'r') as myfile:
    data=myfile.read().lower()
    #parse file for words only
    numToTake = int(sys.argv[3])
    words = re.findall(r"[\w']+|[.,!?;]", data)[:numToTake]
    obs = pd.Series(words)
uniqueObs = set(words)
#total number of unique possible observations
m = len(uniqueObs)
#matrix of probabilities that each state will output a given word
eprobs = np.ndarray.flatten(np.random.dirichlet(np.ones(m), size=1))
for i in range(numStates-1):
    arr = np.ndarray.flatten(np.random.dirichlet(np.ones(m), size=1))
    eprobs = np.vstack([eprobs, arr])
#matrix of probabilities that each state will transition to a given state
tprobs = np.ndarray.flatten(np.random.dirichlet(np.ones(numStates), size=1))
for i in range(numStates-1):
    arr = np.ndarray.flatten(np.random.dirichlet(np.ones(numStates), size=1))
    tprobs = np.vstack([tprobs, arr])

emit = pd.DataFrame(eprobs, index=states, columns=uniqueObs)
trans = pd.DataFrame(tprobs, index=states, columns=states)

print("original trans", trans)

# HMM ALGORITHMS

def forward(prior, trans, emit, states, obs):
    # create trellis like trellis[t][state] e.g. trellis[0]["s1"]
    #structure to calculate probability of given output from given [time][state] tuple using dynamic programming
    trellis = pd.DataFrame(np.zeros([len(states),len(obs)]), index=states, columns=range(len(obs)))
    
    for t in trellis.columns[:-1]:
        if t == 0:
            #at t=0, forward prob to be in a state is (prob of starting in state * prob of obs(0))
            trellis.loc[:,t] = np.log(prior.loc[:,"prob"]) + np.log(emit.loc[:, obs[t]])
        
        #trellis.loc[:, t+1] = np.log(emit.loc[:, obs[t+1]]) + logsumexp(trellis.loc[:,t] + np.log(trans.loc[:,i_state]))
        #fill in probs for the next timeslot based on prob of this timeslot
        #probability to be in a given state at t+1 is
        #probability of getting to that state at time t+1 * prob of emitting obs(t+1) from that state
        for i_state in states:
            
            log_summation = logsumexp(trellis.loc[:,t] + np.log(trans.loc[:,i_state]))
            
            trellis.loc[i_state, t+1] = np.log(emit.loc[i_state, obs[t+1]]) + log_summation
    
    
    # get total prob
    # sum the last column across all states
    # total prob is used to check if backward and forward algorithms agree for convergence
    total_prob = logsumexp(trellis.loc[:,trellis.columns[-1:]].values)
    
    return trellis, total_prob
      
# BACKWARD
def backward(prior, trans, emit, states, obs):
    # create trellis like trellis[t][state] e.g. trellis[0]["s1"]
    #mostly same algo as forward but reversed
    trellis = pd.DataFrame(np.zeros([len(states),len(obs)]), index=states, columns=range(len(obs)))
    for t in reversed(trellis.columns):
        if t == trellis.columns[-1]:
            #backward probability of being in any state at the end is 1
            trellis.loc[:,t] = np.log(1)
        else:
            for i_state in states:
                
                log_summation = logsumexp(trellis.loc[:,t+1] + np.log(trans.loc[i_state,:]) + np.log(emit.loc[:, obs[t+1]]))
                
                trellis.loc[i_state, t] = log_summation
        
    return trellis

def calcTrans(arr, i ,j):
    denom = logsumexp(arr[:,i,:].flatten())
    num = logsumexp(arr[:,i,j].flatten())
    return np.exp(num-denom)
def calcEmit(arr, i, k):
    denom = logsumexp(arr[:,:,i].flatten())
    numeratorComponents = np.array(np.log(0)) 
    #get list of all timeslots where this output was emitted
    t = list(obs[obs==k].index)
    ksiT = arr[t]
    numeratorComponents = np.append(numeratorComponents, (ksiT).reshape(-1, ksiT.shape[-1])[:,i])
    # sum the list to get the numerator value
    num = logsumexp(numeratorComponents.flatten())
    return np.exp(num - denom)

# FORWARD-BACKWARD Baum Welch algorithm
def fwd_bwd():
    global emit, trans
    # get forward trellis and total prob of trellis
    f_trellis, total_prob = forward(prior, trans, emit, states, obs)
    # get backward trellis
    b_trellis = backward(prior, trans, emit, states, obs)
    # calculate Xi (aka ksi)
    # Probability of transiting from to Si to Sj at time t given output O
    # make 3D ksi array ksi_arr[t,i,j]
    ksi_arr = np.array(np.zeros([len(obs),len(states), len(states)]))
    print("Completed forward-backward. Updating matrices.")
    # fill the ksi_arr
    for t in range(len(obs)-1):
        for i, i_state in enumerate(states):
            ksi_arr[t,i,:] = f_trellis.loc[i_state,t] + np.log(trans.loc[i_state, :]) + np.log(emit.loc[:, obs[t+1]]) + b_trellis.loc[:, t+1] - total_prob
    # update trans matrix A         
    trans = pd.DataFrame([[calcTrans(ksi_arr,i,j) for j in range(len(states))] for i in range(len(states))], 
                        index=states, columns=states)
    # update emit matrix B            
    emit = pd.DataFrame([[calcEmit(ksi_arr,i,k) for k in emit.columns] for i in range(len(states))],
                    index=states, columns=uniqueObs)
    # return total_prob so that we can test for convergence between consecutive runs
    return total_prob

# initialize probs far apart
last_total_prob = np.log(0)
total_prob = np.log(1)
def isDone(last_total_prob, total_prob, iteration):
    #relax convergence requirements as we take longer so it doesn't take forever
    if iteration>40: return True
    probDif = np.abs(last_total_prob-total_prob)
    if iteration<=10:
        return probDif<1
    else:
        return probDif<5
# TRAINING
# run fwd-bwd algo until the probabilies converge
iteration = 0
while not isDone(last_total_prob, total_prob, iteration):
    print("Difference of total probabilities:", np.abs(last_total_prob - total_prob))
    iteration+=1
    print("Beginning iteration #", iteration, "of Baum-Welch forward-backward.")
    last_total_prob = total_prob
    total_prob = fwd_bwd()
print("Difference of total probabilities:", np.abs(last_total_prob - total_prob))

#save the final outputs
#this is the model used by genfromtrained.py to generate words
print("Total probability has reached convergence threshold. Writing model probabilities to trans_final, emit_final, and prior.")
trans_final_out = open("trans_final.csv","w")
trans_final_out.write(trans.to_csv())
trans_final_out.close()

emit_final_out = open("emit_final.csv","w")
emit_final_out.write(emit.to_csv())
emit_final_out.close()

prior_final_out = open("prior.csv", "w")
prior_final_out.write(prior.to_csv())
prior_final_out.close()



