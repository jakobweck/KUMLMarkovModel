# import matplotlib.pyplot as plt
# from matplotlib import style
import numpy as np
# from scipy.stats import norm
# import argparse
import pandas as pd
# import random
import sys
import os
from pprint import pprint
import random
import re
def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
    return edges


def randomItemFromWeightedDict(dct):
    rand_val = random.random()
    total = 0
    for k, v in dct.items():
        total += v
        if rand_val <= total:
            return k
    assert False, 'unreachable'

def genText(words, outputLen, wordsToTake):
    statePairs = {}
    wordCounts = {}
    #dict of dicts
    #pair each word in the corpus with a dict mapping each possible successor word to its num of occurrences
    for i, word in enumerate(words[1:wordsToTake], 1):
        #also count occurrences of each word while we're at it
        if not words[i-1] in wordCounts:
            wordCounts[words[i-1]] = 1
        else:
            wordCounts[words[i-1]] += 1

        if not words[i-1] in statePairs:
            statePairs[words[i-1]] = {}
        if not words[i] in statePairs[words[i-1]]:
            statePairs[words[i-1]][words[i]] = 1
        else:
            statePairs[words[i-1]][words[i]] = statePairs[words[i-1]][words[i]]+1
    #divide successor occurrences by total occurrences to get probabilities of each successor
    for word in statePairs:
        totalCount = 0
        for suc in statePairs[word]:
            totalCount += statePairs[word][suc]
        for suc in statePairs[word]:
            statePairs[word][suc] /= totalCount
    totalCount = 0
    for word in wordCounts:
        totalCount += wordCounts[word]
    for word in wordCounts:
        wordCounts[word] /= totalCount

    output = ""
    #get random starting word based on frequency of all words
    startingWord = randomItemFromWeightedDict(wordCounts)
    currWord = (startingWord, statePairs[startingWord])
    #randomly generate 100-word predictive text
    for i in range(outputLen):
        noSpace = (currWord[0]=='?' or currWord[0]==',' or currWord[0]=='.' or currWord[0]=='!')
        if noSpace:
            #remove spaces before punctuation
            output = output[:-1]
        output += (currWord[0]) + " "
        nextWord = randomItemFromWeightedDict(currWord[1])
        currWord = (nextWord, statePairs[nextWord])
    return (output)


def main():
    inputFileName = sys.argv[1]
    with open(inputFileName, 'r') as myfile:
        data=myfile.read().lower()
        #parse file for words only
        words = re.findall(r"[\w']+|[.,!?;]", data)
        print("Enter 1 to generate text, 2 to predict.")
        opt = int(input(":"))
        if opt==1:
            wordsToTake = input("How many words to take from the file? Lots of words might take a while: ")
            numWords = input("How many words to generate?: ")
            text = genText(words, int(numWords), int(wordsToTake))
            print(text)
        else:
            print("Nope")
main()