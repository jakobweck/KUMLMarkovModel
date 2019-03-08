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
    print(currWord)
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


print("Enter 1 to generate text, 2 to predict.")
opt = int(input(":"))
if opt==1:
    numWords = input("How many words to generate?: ")
    text = genText(int(numWords))
    print(text)
else:
    print("Nope")

