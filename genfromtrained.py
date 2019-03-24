import pandas as pd
import numpy as np
from scipy.special import logsumexp
import re
import sys
import random

def randomLabelFromProbabilitySeries(ser):
    rand_val = random.random()
    total = 0
    for index, prob in ser.items():
        total += prob
        if rand_val <= total:
            return index
    assert False, 'unreachable'

def randomLabelFromNonNormalProbabilitySeries(ser):
    rand_val = random.random()
    total = 0
    normalizer = 1/sum(ser.values)
    for index, prob in ser.items():
        total += (prob * normalizer)
        if rand_val <= total:
            return index
    assert False, 'unreachable'

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

# def verterbi(pi, a, b, obs):
#     nStates = np.shape(b)[0]
#     T = np.shape(obs)[0]

#     # init blank path
#     path = np.zeros(T)
#     # delta --> highest probability of any path that reaches state i
#     delta = np.zeros((nStates, T))
#     # phi --> argmax by time step for each state
#     phi = np.zeros((nStates, T))

#     # init delta and phi
#     delta[:, 0] = pi * b[:, obs[0]]
#     phi[:, 0] = 0

#     print('\nStart Random Walk Forward\n')
#     # Forward Algorithm Extension
#     for t in range(1, T):
#         for s in range(nStates):
#             delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]]
#             phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
#             print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s,t]))

#     # Find optimal path
#     print('-'*50)
#     print('Start Backtrace\n')
#     path[T-1] = np.argmax(delta[:, T-1])

#     for t in range(T-2, -1, -1):
#         a = int(path[t+1])
#         b = [t+1]
#         path[t] = phi[a, b]
#         print('path[{}] = {}'.format(t, path[t]))

# return path, delta, phi

emit = pd.read_csv("emit_final.csv",index_col=0)
#transition probabilities from states to states
trans = pd.read_csv("trans_final.csv",index_col=0)
#probability of starting in each state
prior = pd.read_csv("prior.csv",index_col=0)

#to predict outputs/generate outputs
#train model(run fwd_bwd until convergence)
#start in a random state based on prior
#output a word randomly using the probabilities in that state's emission probs
#change to another state (or stay in the same) randomly using the probabilities in that state's transition probs
def genText(numWords):
    output = ""
    initStateProbs = prior.loc[:, "prob"]
    currState = randomLabelFromProbabilitySeries(initStateProbs)
    currTransRow = trans.loc[currState,:]
    currEmitRow = emit.loc[currState,:]
    currWord = randomLabelFromProbabilitySeries(currEmitRow)
    for i in range(numWords):
        noSpace = (currWord[0]=='?' or currWord[0]==',' or currWord[0]=='.' or currWord[0]=='!')
        if noSpace:
            #remove spaces before punctuation
            output = output[:-1]
        output += (currWord) + " "
        currState = randomLabelFromProbabilitySeries(currTransRow)
        currTransRow = trans.loc[currState,:]
        currEmitRow = emit.loc[currState,:]
        currWord = randomLabelFromProbabilitySeries(currEmitRow)
    return output

def CheckIfWordWork(words) :
    words = words.split()
    match = False
    wordList = []

    with open("emit_final.csv") as emit:
        lis = [line.split(',') for line in emit]
        wordList = lis[0]

    # Are these words part of the dataframe we created?
    # Let's check
    for word in words :
        match = False
        for st in wordList:
            if(st == word) :
                match = True
                continue
        if match == False :
            print("There was no matching candidate for this word: ", word)
            return False
    
    # The words are all valid, continue to viterbi prediction?
    print("All words entered were valid, continuing to prediction...")
    return True

def predict(words):
    words = re.findall(r"[\w']+|[.,!?;]", words)
    initStateProbs = prior.loc[:, "prob"]
    print("Words: ", words)
    currWord = words[0]
    wordStateProbs = emit.loc[:, currWord]
    # Get a random state, weighted toward the most likely state that produced this word
    currState = randomLabelFromNonNormalProbabilitySeries(wordStateProbs)
    currTransRow = trans.loc[currState,:]
    print("CURRENT STATE: ", currState)
    # print("Current Trans Row: ", currTransRow)

    wordStateProbsArray = []

    for worIter in range(len(words)):
        wordStateProbsArray.append(emit.loc[:, words[worIter]])
    print(wordStateProbsArray)

    with open("Shakespeare_data.csv", 'r') as myfile:
        data=myfile.read().lower()
        #parse file for words only
        words = re.findall(r"[\w']+|[.,!?;]", data)[:10000]
        obs = pd.Series(words)

    states = emit.index

    # Obs is the observations of behavior

    viterbi_path = viterbi(prior, trans, emit, states, obs)
    Debug.Log("Viterbi Path: ", viterbi_path)
    # path, delta, phi = viterbi(wordStateProbsArray, a, b, obs)
    # print('\nsingle best state path: \n', path)
    # print('delta:\n', delta)
    # print('phi:\n', phi)
    
    
print("Enter 1 to generate text, 2 to predict.")
opt = int(input(":"))
if opt==1:
    numWords = input("How many words to generate?: ")
    text = genText(int(numWords))
    print(text)
else:
    words = input("Input a string from the dataset:")
    if (CheckIfWordWork(words)) :
        predict(words)

