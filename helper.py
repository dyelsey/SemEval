#!/usr/bin/env python

from operator import itemgetter

"""
conflates 'objective' into neutral
"""
def conflate (tweetData, trainingExList, conf):
    dictSent = {}
    neg = 0.0
    pos =0.0
    neutral =0.0
    obj =0.0
    objNeut =0.0

    for t in trainingExList:
        if len(tweetData['tweets'][t]['answers']) == 2:
            objNeut +=1
        elif 'negative' in tweetData['tweets'][t]['answers']:
            neg += 1
        elif 'positive' in tweetData['tweets'][t]['answers']:
            pos += 1
        elif 'neutral' in tweetData['tweets'][t]['answers']:
            neutral +=1
        elif 'objective' in tweetData['tweets'][t]['answers']:
            obj += 1

    total = (neg+pos+neutral+obj+objNeut)

    dictSent['negative'] = neg, (neg/total)*100
    dictSent['positive'] = pos, (pos/total)*100
    if conf:
        dictSent['neutral'] = (neutral+obj+objNeut), \
                100*(neutral+obj+objNeut)/total
    else:
        dictSent['neutral'] = neutral, (neutral/total)*100
        dictSent['objective'] = obj , (obj/total)*100
        dictSent['objective-OR-neutral'] = objNeut , (objNeut/total)*100

    return dictSent


#finds accuracy if we randomly guess
def randomGuess(trainingExList, dictSent):
    acc = 0.0
    totalInstance = 0.0
    numEx = len(trainingExList)
    a = (1.0/len(dictSent))*numEx
    acc = a/numEx
    return acc

"""
finds the most frequent semantic
"""
def MFS(dictSent):
    mfsense = max(dictSent.iteritems(), key=itemgetter(1))[0]
    return mfsense
