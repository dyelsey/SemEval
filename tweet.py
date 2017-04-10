#!/usr/bin/env python

import crossval
from parseTweet import parse_tweets
from operator import itemgetter
from declist import makeDecList, getFeatureCounts

import sys
from arktweet import tokenize
import naivebayes

"""
stores results for each instance(tweet)
"""
def calcResults(decList, listIDTest, mfs, testData):
    results = {}
    for instance2 in listIDTest: #Testing instances
        featC = {}
        featC = getFeatureCounts(instance2, testData) #testinstance features
        for d in decList:
            if d[0][0] in featC:#if feature of decList in instance feature
                if d[1] > 0: #reach end of declist for score > 0
                    results[instance2] = d[0][1] #assign its sense
                else:
                    results[instance2] = mfs[0]
                break
            else: #reach end of declist
                results[instance2]=mfs[0]
    return results

"""
checks accuracy for decList
"""
def accuracy(i, testData, results):
    acc = 0.0
    totalAcc = 0.0
    for l in results:
        if l in testData['tweets'].keys():
            if results[l] in testData['tweets'][l]['answers']:
                acc+=1

    totalAcc+= acc
    print 'Total accuracy is {0:.2f}% with chunk {1} as test'.format(totalAcc/len(results)*100, i)

if __name__=='__main__':
    K = 5 #number of cross-validation chunks
    CONF = 1    #Conflate
    token = 1   #Tokenize or not
    useDec = 0  #use decList or naive bayes


    filename = sys.argv[1]
    tweetData = parse_tweets(filename, 'B')
    if token:
        inst_ids = tweetData['tweets'].keys()
        tweets = [' '.join(tweetData['tweets'][i]['words']) for i in inst_ids]
        tok = tokenize(tweets)
        for j in range(len(inst_ids)):
            i = inst_ids[j]
            tok[j] = tok[j].split()
            tweetData['tweets'][i]['words'] = tok[j]

    trainingExList = tweetData['tweets'].keys()

    chunks = crossval.makeChunks(trainingExList, K)
    toTest ={}

    if useDec:
        print "With Decision List:"
    else:
        print "With Naive Bayes:"
    for i in range(K): #To use each chunk has a test
        toTest[i] = crossval.createSet(i, chunks)
        trainID = toTest[i]['train']
        mfs = crossval.mfsChunk(tweetData, chunks, CONF)
        testID = toTest[i]['test']
        if useDec:
            decList = makeDecList(tweetData, trainID)
            results = calcResults(decList, testID, mfs, tweetData)
            accuracy(i, tweetData, results)
        else:
            naivebayes.makeNaiveBayes(tweetData, trainID, testID, CONF, i)
