#!/usr/bin/env python

from parseTweet import parse_tweets
from operator import itemgetter
import helper
from collections import defaultdict

"""
makes the various chunks, depending on number of chunks K
"""
def makeChunks(trainingExList, K):
    numChunk = (len(trainingExList) / K)+1
    dictChunk = []
    chunkCounter = 0
    for i in range(K):
        dictChunk.append(trainingExList[chunkCounter:chunkCounter+numChunk])
        chunkCounter+=numChunk
    return dictChunk

"""
finds most common sentiment per chunk
"""
def mfsChunk(tweetData, dictChunk, conf):
    mfsChunks = {}
    for chunk in range(len(dictChunk)):
        dictSent = helper.conflate(tweetData, dictChunk[chunk], conf)
        mostFreq = helper.MFS(dictSent)
        mfsChunks[chunk] = mostFreq
    return mfsChunks

"""
Creates a 'test' and 'train' data from the chunks, saved in dictionary
"""
def createSet(i, dictChunk):
    training = []
    test = []
    for j in range(len(dictChunk)):
        if i != j:
            training += dictChunk[j]
        else:
            test += dictChunk[i]
    toReturn = {}
    toReturn['train'] = training
    toReturn['test'] = test
    return toReturn

"""
Tests accuracy of most frequent sentiment
"""
def testAcc(dictChunk, K, tweetData, CONF):
    accTest = {}
    for i in range(K):
        sets = createSet(i, dictChunk)
        s = []
        s.append(sets['train'])
        trainMFS = mfsChunk(tweetData, s, CONF)

        dictSent = helper.conflate(tweetData, sets['test'], CONF)
        accTest[i] = dictSent[trainMFS[0]][1]
        acc = 0.0
    return accTest
