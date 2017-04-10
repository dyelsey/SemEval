#!/usr/bin/env python

"""
Decision list classifier
"""

from operator import itemgetter
from math import log
from collections import defaultdict

"""
Returns dictionary of feature, counts for one tweet(instance)
"""
def getFeatureCounts(instance, trainData):
    featCount = defaultdict(int)
    no = set(['not'])
    punc = set([',', '.','?','!', ';', ':', "'", '"'])
    words =  trainData['tweets'][instance]['words']
    mostCommon = set(["and","the","I",".","a",",","of","to","in","for"])
    negate = False
    ctr = 0

    for i in range(len(words)):
        if words[i] in no:
            if negate == False:
                negate = True
            else:
                negate = False
        if words[i] in punc and negate == True:
            negate = False

        words[i] = words[i].lower() #casefolding

        #if words[i] not in mostCommon: #stopwords
        if negate:
            words[i] = 'NOT_'+words[i]

            #s = "" #bigrams makes it worse!
            #if i < len(words)-1:
            #    s = words[i]+"_"+words[i+1]

        featCount[words[i]] +=1
            #featCount[s] +=1

    return featCount

"""
Update senseCounts which is a dictionary of sense, dictionary(feature,count)
"""
def updateSense(senses, featCounts, senseCounts):
    for sense in senses:
        for feature in featCounts:
            senseCounts[sense][feature] += featCounts[feature]
    return senseCounts

"""
Calculates score of pair feature,sense
"""
def calcScore(feature, sense, senseCounts):
    ALPHA = .1 #smoothing
    num = senseCounts[sense][feature]
    num += ALPHA
    den = 0.0
    for s in senseCounts:
        if s != sense:
            if feature in senseCounts[s]:
                den += senseCounts[s][feature]
    den += ALPHA
    score = log((num/den),2)
    return score

"""
Calculates most frequent Sentiment given word
"""
def makeDecList(trainData, listIDTrain):
    decList = {}
    senseCounts = defaultdict(lambda:defaultdict(int))
    for instance in listIDTrain: #Training instances
        featCounts = getFeatureCounts(instance, trainData)
        sense = trainData['tweets'][instance]['answers']
        senseCounts = updateSense(sense, featCounts, senseCounts)
    for s in senseCounts:
        counter = 0
        for feature in senseCounts[s]:
            pair = feature,s
            score = calcScore(feature, s, senseCounts)
            if score > 0:
                decList[pair] = score
    decList = sorted(decList.items(), key= itemgetter(1), reverse=True)
    return decList
