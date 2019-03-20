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
    
    # print(wordList)
    # print(wordList[5])
    # return

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
    currWord = words[0]
    wordStateProbs = emit.loc[:, currWord]
    #get a random state, weighted toward the most likely state that produced this word
    currState = randomLabelFromNonNormalProbabilitySeries(wordStateProbs)
    currTransRow = trans.loc[currState,:]
    
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

