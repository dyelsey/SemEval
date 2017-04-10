#!/usr/bin/env python
import declist
import helper
from collections import defaultdict

"""
Naive Bayes classifier
"""

def makeNaiveBayes(tweetData, listIDTrain, listIDTest, CONF, i):
    dictSent = helper.conflate(tweetData, listIDTrain, CONF)
    featCount = {}
    testCounts = {}
    ALPHA = .1 #smoothing
    senseCounts = defaultdict(lambda:defaultdict(int))
    acc = 0.0
    senseDenom = {}
    for instance in listIDTest: #Test instances
        testCounts = declist.getFeatureCounts(instance, tweetData)

    for instance in listIDTrain: #Training instances
        featCount = declist.getFeatureCounts(instance, tweetData)
        sense = tweetData['tweets'][instance]['answers']
        sense = ifConf(sense, CONF)
        senseCounts = updateSense(sense, featCount, senseCounts, CONF)
    v = set()
    v.update(testCounts.keys())
    v.update(featCount.keys())
    vSize = len(v) #number of types in training and test
    smoothing = vSize*ALPHA #smoothing
    neg = 0.0
    for tweet in listIDTest:
        maxv = -1
        maxs = ""
        ct = 0
        total = 0
        for sense in senseCounts:
            senseDenom[sense] = calcDenom(senseCounts, sense)
            product = dictSent[sense][0]/len(listIDTrain)
            for feature in  declist.getFeatureCounts(tweet, tweetData):
                total += 1
                prob = (senseCounts[sense][feature]+ALPHA) / \
                        (senseDenom[sense]+smoothing)
                product *= prob #complete naive bayes formula
            if product > maxv:
                maxv = product
                maxs = sense

        #check if sense correct for tweet
        correctSense = ifConf(tweetData['tweets'][tweet]['answers'], CONF)
        if maxs in correctSense:
            acc+=1.0 #increments if correct
    perc = (acc/len(listIDTest))*100
    print "Accuracy with chunk {0} as testing: {1:.2f}%".format(i, perc)

"""
Calculate the denominator for the formula of naive bayes
"""
def calcDenom(senseCounts, sense):
    den = 0.0
    for feature in senseCounts[sense]:
        den += senseCounts[sense][feature]
    return den

"""
Conflates the sentiments
"""
def ifConf(senses, CONF):
    if CONF:
        if len(senses) == 2:
            return ["neutral"]
        if senses[0] == 'objective':
            return ['neutral']
    return senses

"""
Creates a dictionary of all the sentiments and how frequent their features are
"""
def updateSense(sense, featCounts, senseCounts, CONF):
    for feature in featCounts:
        for s in sense:
            senseCounts[s][feature] += featCounts[feature]
    return senseCounts
