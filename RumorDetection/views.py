
from django.shortcuts import render
from django.http import HttpResponse

import pickle
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
import json, os
import pandas as pd
from os import listdir
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.layers import Flatten,TimeDistributed,MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.utils import np_utils
from keras.optimizers import Adagrad
import gensim
import check_code
import os
import tweepy
import json
import matplotlib.pyplot as plt

access_token = '1185203655705653249-MI55LF0YDXJHU06sATMMQtHjfWrkKF'  # access details of tweet
access_token_secret = '9PCjIq70gj0o8zHgrtkvr1K6vlTo5H2P1TyQcfOqyh7B3'
consumer_key = 'kkEHLGQM2iRKAIRzh33eAQ2RN'
consumer_secret = 'pNa2PFNYUVWSnUPYuXaFPybExsl3ABY1oLHBkPTtY1AZIJpFd7'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
# Create your views here.

mypath = "check"
Evenlist = listdir(mypath)
TIME_STEPS = 6
InputWord = 15
IMPUT_SIZE = 300 * InputWord
BATCH_SIZE = 30
BATCH_INDEX = 0
OUTPUT_SIZE = 2
CELL_SIZE = 175
LR = 0.001

def ContinuousInterval(intervalL):  # Returns max continuous interval
    maxInt = []
    tempInt = [intervalL[0]]
    for q in range(1, len(intervalL)):
        if intervalL[q] - intervalL[q - 1] > 1:
            if len(tempInt) > len(maxInt):
                maxInt = tempInt
            tempInt = [intervalL[q]]
        else:
            tempInt.append(intervalL[q])
    if len(maxInt) == 0:
        maxInt = tempInt
    return maxInt

def remove_link(input_text):  # remove hyperlink
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    tweetText = pattern.sub(' ', input_text)
    return tweetText


def remove_pattern(input_txt, pattern):  # remove handles
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, ' ', input_txt)
    return input_txt


def remove_spchar(input_text):  # remove digits
    tweetText1 = re.sub('[0-9]', ' ', input_text)
    return tweetText1


def remove_emoji(input_text):  # remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' ', input_text)


def index(request):
    #return httpResponse file
    return render(request, 'index.html')


def home(request):
    return render(request, 'home.html')

def data(request):
    if request.method == "GET":
        tweet = request.GET['tweet']
        count = request.GET['count']
        count = int(count)

    print(tweet)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    # api = tweepy.API(auth)
    text1 = []
    credentials = {}

    for tweet1 in tweepy.Cursor(api.search, q=str(tweet), lang="en").items(count):
        credentials['text'] = tweet1.text
        text1.append(tweet1.text)
        Time = tweet1.created_at
        Time = int(Time.strftime("%Y%m%d%H%M%S"))
        credentials['time'] = Time
        var = tweet1.id

        #with open(os.path.join("check\event", str(var) + ".json"), "w") as file:
        #   json.dump(credentials, file)

    print("printing..")
    for tweet1 in text1:
        print(tweet1)

    content1 = {
        'text1': text1
    }
    return render(request, 'data.html', content1)

def graph(request):
    return render(request, 'graph.html')


def result(request):

    result = check_code.return_val()
    print(result)
    content = {
        'result': result
    }
    return render(request, 'result.html', content)