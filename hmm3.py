# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:53:23 2016

@author: Emilio Esposito
All probabilities calculated use log-probabilities, EXCEPT for the inital dfs: emit, trans, prior
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
#transition probabilities from states to states
#probability of starting in each state
priors = np.ndarray.flatten(np.random.dirichlet(np.ones(numStates), size=1))
prior = pd.DataFrame(priors, index = states, columns = ["prob"])
#observations
filename = ""
if os.name == 'nt':
    filename = (os.getcwd()+ "\\" + sys.argv[1])
else:
    filename = (sys.argv[1])

with open(filename, 'r') as myfile:
    data=myfile.read().lower()
    #parse file for words only
    words = re.findall(r"[\w']+|[.,!?;]", data)[:1000]
    obs = pd.Series(words)
uniqueObs = set(words)
m = len(uniqueObs)
eprobs = np.ndarray.flatten(np.random.dirichlet(np.ones(m), size=1))
for i in range(numStates-1):
    arr = np.ndarray.flatten(np.random.dirichlet(np.ones(m), size=1))
    eprobs = np.vstack([eprobs, arr])

tprobs = np.ndarray.flatten(np.random.dirichlet(np.ones(numStates), size=1))
for i in range(numStates-1):
    arr = np.ndarray.flatten(np.random.dirichlet(np.ones(numStates), size=1))
    tprobs = np.vstack([tprobs, arr])

emit = pd.DataFrame(eprobs, index=states, columns=uniqueObs)
trans = pd.DataFrame(tprobs, index=states, columns=states)

states = emit.index

print("original trans", trans)

# HMM ALGORITHMS

# FORWARD
def forward(prior, trans, emit, states, obs):
    # create trellis like trellis[t][state] e.g. trellis[0]["s1"]
    trellis = pd.DataFrame(np.zeros([len(states),len(obs)]), index=states, columns=range(len(obs)))
    
    for t in trellis.columns[:-1]:
        if t == 0:
            trellis.loc[:,t] = np.log(prior.loc[:,"prob"]) + np.log(emit.loc[:, obs[t]])
        
#        trellis.loc[:, t+1] = np.log(emit.loc[:, obs[t+1]]) + logsumexp(trellis.loc[:,t] + np.log(trans.loc[:,i_state]))
        for i_state in states:
            
            log_summation = logsumexp(trellis.loc[:,t] + np.log(trans.loc[:,i_state]))
            
            trellis.loc[i_state, t+1] = np.log(emit.loc[i_state, obs[t+1]]) + log_summation
    
    
    # get total prob
    # sum the last column across all states
    total_prob = logsumexp(trellis.loc[:,trellis.columns[-1:]].values)
    
    return trellis, total_prob
      
# BACKWARD
def backward(prior, trans, emit, states, obs):
    # create trellis like trellis[t][state] e.g. trellis[0]["s1"]
    trellis = pd.DataFrame(np.zeros([len(states),len(obs)]), index=states, columns=range(len(obs)))
    for t in reversed(trellis.columns):
        if t == trellis.columns[-1]:
            trellis.loc[:,t] = np.log(1)
        else:
            for i_state in states:
                
                log_summation = logsumexp(trellis.loc[:,t+1] + np.log(trans.loc[i_state,:]) + np.log(emit.loc[:, obs[t+1]]))
                
                trellis.loc[i_state, t] = log_summation
    
    # get total prob
    total_prob = logsumexp(np.log(prior.loc[:,"prob"]) + np.log(emit.loc[:, obs[0]]) + trellis.loc[:, 0])
    
    return trellis

        
# VITERBI
def viterbi(prior, trans, emit, states, obs):
    # create trellis like trellis[t][state] e.g. trellis[0]["s1"]
    trellis = pd.DataFrame(np.zeros([len(states),len(obs)]), index=states, columns=range(len(obs)))
    
    path = pd.DataFrame(index = states, columns=range(len(obs)))
       
    for t in trellis.columns[:-1]:
    
        if t == 0:
            trellis.loc[:,t] = np.log(prior.loc[:,"prob"]) + np.log(emit.loc[:, obs[t]])
            path.loc[:, t] = states
        
        for i_state in states:
            
            max_prob = np.max(trellis.loc[:,t] + np.log(trans.loc[:,i_state]))
            
            trellis.loc[i_state, t+1] = np.log(emit.loc[i_state, obs[t+1]]) + max_prob
            path.loc[i_state, t+1] = trellis.loc[:,t+1].idxmax()
    
    max_state = trellis.iloc[:,-1].idxmax()
            
    return path.loc[max_state]

# FORWARD-BACKWARD Baum Welch algorithm
def fwd_bwd():
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
   #    trans = [doThing(ksi_arr,i,j) for i in enumerate(states) for j in enumerate(states)]   
    for i, i_state in enumerate(states):
       trans_denominator = logsumexp(ksi_arr[:,i,:].flatten())
       for j, j_state in enumerate(states):
           trans_numerator = logsumexp(ksi_arr[:,i,j].flatten())
           # update trans probs and convert from log prob to normal prob by using exponentiation
           trans.loc[i_state, j_state] = np.exp(trans_numerator - trans_denominator)
    # update emit matrix B            
    for j, j_state in enumerate(states):
        emit_denominator = logsumexp(ksi_arr[:,:,j].flatten())
       
       # loop through all k emission output possibilities
        for k in emit.columns:
            emit_numerator_arr = np.array(np.log(0)) 
          #  emit_slow = np.array(np.log(0))
            # sum ksi where k == observed output and store it in numerator
            t = list(obs[obs==k].index)
            # for i in t:
            #     emit_slow = np.append(emit_slow, ksi_arr[i,:,j].flatten())
            ksiT = ksi_arr[t]
            emit_numerator_arr = np.append(emit_numerator_arr, (ksiT).reshape(-1, ksiT.shape[-1])[:,j])
            # sum the list to get the numerator value
            emit_numerator = logsumexp(emit_numerator_arr.flatten())
            
            # update emission probs and convert from log prob to normal prob by using exponentiation
            emit.loc[j_state, k] = np.exp(emit_numerator - emit_denominator)
    # return total_prob so that we can test for convergance between consecutive runs
    return total_prob

# initialize probs far apart
last_total_prob = np.log(0)
total_prob = np.log(1)

# TRAINING
# run fwd-bwd algo until the probabilies converge
iteration = 0
while np.abs(last_total_prob - total_prob) > 1:
    print(np.abs(last_total_prob - total_prob))
    iteration+=1
    print("Beginning iteration #", iteration, "of Baum-Welch forward-backward.")
    last_total_prob = total_prob
    total_prob = fwd_bwd()

# DECODING
viterbi_path = viterbi(prior, trans, emit, states, obs)

# save the final outputs
print("Writing output to trans_final, emit_final, and viterbi_path_final.")
trans_final_out = open("trans_final.csv","w")
trans_final_out.write(trans.to_csv())
trans_final_out.close()

emit_final_out = open("emit_final.csv","w")
emit_final_out.write(emit.to_csv())
emit_final_out.close()

viterbi_path_final_out = open("viterbi_path_final.csv","w")
viterbi_path_final_out.write(pd.DataFrame(viterbi_path).to_csv())
viterbi_path_final_out.close()

prior_final_out = open("prior.csv", w)
prior_final_out.write(prior.to_csv())
prior_final_out.close()



